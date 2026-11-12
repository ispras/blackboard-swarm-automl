import numpy as np, pandas as pd, json, os, sys
sys.path.insert(0, '.')
from postprocess_search import _bounds, _roll_mean_center

R='.'; A='../achernar_sigma'; C='../castor_alpha'
GKEYS=['game_play','nfl_player_id_1','nfl_player_id_2']

og=pd.read_parquet('oof_ground.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step','contact','fold']]
gid=og.groupby(GKEYS,sort=False).ngroup().values.astype(np.int64)
print('gid monotonic',bool(np.all(np.diff(gid)>=0)))
gs,ge=_bounds(gid)
yg=og.contact.values.astype(np.int8)

models={}
def add(name,oop,ocol,tp,tcol):
    try:
        o=og[['contact_id']].merge(pd.read_parquet(oop)[['contact_id',ocol]],on='contact_id',how='left')[ocol].values.astype(np.float64)
        t=pd.read_parquet(tp)[['contact_id',tcol]].rename(columns={tcol:name})
        models[name]=(o,t)
        print('model',name,'oof',len(o),'nan',int(np.isnan(o).sum()),'test',len(t))
        return True
    except Exception as e:
        print('skip',name,e); return False

SPECS={
 'track':('oof_ground.parquet','prob','testpred_ground.parquet','prob'),
 'ach2':(f'{A}/oof_ground_lgbv2.parquet','pred',f'{A}/testpred_ground_lgbv2.parquet','pred'),
 'gfus':(f'{A}/oof_ground_gfus.parquet','pred',f'{A}/testpred_ground_gfus.parquet','pred'),
 'ctab':(f'{C}/ground_oof_tab.parquet','oof',f'{C}/ground_test_tab.parquet','pred'),
 'cvid':(f'{C}/ground_oof_video.parquet','pred',f'{C}/ground_test_video.parquet','pred'),
 'ceff':(f'{C}/ground_oof_video_eff.parquet','pred',f'{C}/ground_test_video_eff.parquet','pred'),
 'eb3':(f'{C}/ground_oof_video_eb3.parquet','pred',f'{C}/ground_test_video_eb3.parquet','pred'),
 'cnx2':(f'{C}/ground_oof_video_cnx2.parquet','pred',f'{C}/ground_test_video_cnx2.parquet','pred'),
 'wide':(f'{C}/ground_oof_video_wide.parquet','pred',f'{C}/ground_test_video_wide.parquet','pred'),
 'tight':(f'{C}/ground_oof_video_tight.parquet','pred',f'{C}/ground_test_video_tight.parquet','pred'),
 'r34':(f'{C}/ground_oof_video_r34.parquet','pred',f'{C}/ground_test_video_r34.parquet','pred'),
 'convnext':(f'{C}/ground_oof_video_cnx.parquet','pred',f'{C}/ground_test_video_cnx.parquet','pred'),
 'stackr':('ground_oof_stack_r.parquet','prob','ground_test_stack_r.parquet','prob'),
 'gstack':(f'{A}/oof_ground_gstack.parquet','pred',f'{A}/testpred_ground_gstack.parquet','pred'),
 'gstack2':(f'{A}/oof_ground_gstack2.parquet','pred',f'{A}/testpred_ground_gstack2.parquet','pred'),
}
for n,(a,b,c,d) in SPECS.items():
    if os.path.exists(a) and os.path.exists(c):
        add(n,a,b,c,d)

names=list(models)
O=np.stack([models[n][0] for n in names],axis=1)
# fill any nan with column median
for j in range(O.shape[1]):
    col=O[:,j]; med=np.nanmedian(col)
    O[np.isnan(col),j]=med

def mcc(y,p):
    idx=np.argsort(-p); ys=y[idx].astype(np.float64)
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); pos=ys.sum(); neg=len(ys)-pos
    fn=pos-tp; tn=neg-fp; den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    with np.errstate(divide='ignore',invalid='ignore'):
        m=np.where(den>0,(tp*tn-fp*fn)/den,-1)
    k=int(np.argmax(m)); return float(m[k]), float(np.sort(p)[::-1][k])

def evalw(w,wd):
    p=O@w
    sm=_roll_mean_center(p.astype(np.float32),gs,ge,wd).astype(np.float64)
    return mcc(yg,sm)

rng=np.random.default_rng(7)
best=(-1,None,None,None)
# include castor weights as candidate if all present
try:
    cfg=json.load(open(f'{C}/final_eff_blend.json'))
    w0=np.zeros(len(names))
    for n,ww in zip(cfg['ground_models'],cfg['ground_weights']):
        if n in names: w0[names.index(n)]=ww
    for wd in [7,11,15]:
        r=evalw(w0,wd); print('castor weights wd',wd,round(r[0],5))
        if r[0]>best[0]: best=(r[0],w0.copy(),wd,r[1])
except Exception as e:
    print('castor weights skip',e)

for wd in [7,11,15]:
    b=(-1,None)
    for it in range(700):
        w=rng.dirichlet(np.ones(len(names)))
        r=evalw(w,wd)[0]
        if r>b[0]: b=(r,w)
    print('wd',wd,'best',round(b[0],5),flush=True)
    if b[0]>best[0]: best=(b[0],b[1],wd,None)
print('BEST',best[0],'wd',best[2])
w=best[1].copy(); wd=best[2]; bm=best[0]
for it in range(3):
    for i in range(len(w)):
        for j in range(len(w)):
            if i==j or w[j]<0.01: continue
            w2=w.copy(); w2[i]+=0.01; w2[j]-=0.01
            r=evalw(w2,wd)[0]
            if r>bm: bm=r; w=w2
r=evalw(w,wd)
print('REFINED',r,'weights',dict(zip(names,np.round(w,4))),flush=True)
# save blended oof + test
p_oof=O@w
sm=_roll_mean_center(p_oof.astype(np.float32),gs,ge,wd)
pd.DataFrame({'contact_id':og.contact_id,'game_play':og.game_play,'nfl_player_id_1':og.nfl_player_id_1,
              'nfl_player_id_2':og.nfl_player_id_2,'step':og.step,'contact':yg,'fold':og.fold.values,
              'prob':sm}).to_parquet('ground_oof_best.parquet',index=False)
# test
te=pd.read_parquet('testpred_ground.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']]
for n in names:
    te=te.merge(models[n][1],on='contact_id',how='left')
p_test=sum(w[j]*te[n].values for j,n in enumerate(names))
te['_p']=p_test
# smooth test using same function (sort internally? already sorted by keys? ensure)
te=te.sort_values(GKEYS+['step'],kind='mergesort').reset_index(drop=True)
gid2=te.groupby(GKEYS,sort=False).ngroup().values.astype(np.int64)
gs2,ge2=_bounds(gid2)
sm2=_roll_mean_center(te['_p'].values.astype(np.float32),gs2,ge2,wd)
pd.DataFrame({'contact_id':te.contact_id,'game_play':te.game_play,'nfl_player_id_1':te.nfl_player_id_1,
              'nfl_player_id_2':te.nfl_player_id_2,'step':te.step,'prob':sm2}).to_parquet('ground_test_best.parquet',index=False)
json.dump({'names':names,'weights':[float(x) for x in w],'wd':wd,'ground_mcc':r[0],'thr':r[1]},
          open('ground_best.json','w'),indent=2)
print('saved')
