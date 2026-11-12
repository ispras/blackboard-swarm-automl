import numpy as np, pandas as pd, json, sys
sys.path.insert(0, '.')
from postprocess_search import _bounds, _roll_mean_center

pref=pd.read_parquet('oof_pair.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step','contact','fold']]
# assume saved sorted by keys/step; verify
chk=(pref[['game_play','nfl_player_id_1','nfl_player_id_2','step']].astype(str).agg('|'.join,axis=1))
print('sorted?', chk.is_monotonic_increasing)
yp=pref.contact.values.astype(np.int8)
ids=pref[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']].copy()
gid=ids.groupby(['game_play','nfl_player_id_1','nfl_player_id_2'],sort=False).ngroup().values.astype(np.int64)
gs,ge=_bounds(gid)

def load(path,col):
    return pref[['contact_id']].merge(pd.read_parquet(path)[['contact_id',col]],on='contact_id',how='left')[col].values.astype(np.float64)

names=['track','pairh','ach1','ach2','gstack','seq','xgb']
P=np.stack([
 load('oof_pair.parquet','prob'),
 load('oof_pair_h.parquet','prob'),
 load('../achernar_sigma/oof_pair_lgbv1.parquet','pred'),
 load('../achernar_sigma/oof_pair_lgbv2.parquet','pred'),
 load('../achernar_sigma/oof_pair_gstack.parquet','pred'),
 np.mean([load(f'oof_pair_seq_s{i}.parquet','prob') for i in range(5)],axis=0),
 load('oof_pair_xgb.parquet','prob'),
],axis=1)

def mcc(y,p):
    idx=np.argsort(-p); ys=y[idx].astype(np.float64)
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); pos=ys.sum(); neg=len(ys)-pos
    fn=pos-tp; tn=neg-fp; den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    with np.errstate(divide='ignore',invalid='ignore'):
        m=np.where(den>0,(tp*tn-fp*fn)/den,-1)
    k=int(np.argmax(m)); return float(m[k]), float(np.sort(p)[::-1][k])

def evalw(w, wds=(3,)):
    p=P@w
    out={}
    for wd in wds:
        sm=_roll_mean_center(p.astype(np.float32),gs,ge,wd).astype(np.float64)
        out[wd]=mcc(yp,sm)
    return out

w6=np.array([0.0045405216088693495,0.18332403631402666,0.024654141573700817,0.04354320471882327,0.43074438380762964,0.31319371197695034])
cands=[]
for a in np.arange(0,0.36,0.02):
    cands.append(np.r_[w6*(1-a),a])
rng=np.random.default_rng(0)
for _ in range(120):
    v=rng.dirichlet(np.array([0.4,0.7,0.2,0.3,2.0,1.2,0.6])); cands.append(v)
best=(-1,None)
for w in cands:
    r=evalw(w,(3,))
    if r[3][0]>best[0]: best=(r[3][0],w.copy(),r[3][1])
print('BEST random',best[0],dict(zip(names,np.round(best[1],3))),'thr',best[2],flush=True)

# local refine (coordinate transfers)
w=best[1].copy(); bm=best[0]
for it in range(3):
    for i in range(len(w)):
        for j in range(len(w)):
            if i==j or w[j]<0.01: continue
            w2=w.copy(); d=0.02; w2[i]+=d; w2[j]-=d
            r=evalw(w2,(3,))[3][0]
            if r>bm: bm=r; w=w2
print('REFINED',bm,dict(zip(names,np.round(w,3))),flush=True)
# evaluate best w at w3 and w5
print('final eval',evalw(w,(3,5)))
json.dump({'names':names,'weights':[float(x) for x in w],'mcc_w3':evalw(w,(3,))[3][0],
           'mcc_w5':evalw(w,(5,))[5][0]},open('pair_weights_search.json','w'),indent=2)
# per-fold MCC for chosen w (smoothed w3)
sm=_roll_mean_center((P@w).astype(np.float32),gs,ge,3).astype(np.float64)
for f in range(5):
    m=pref.fold.values==f
    print('fold',f,mcc(yp[m],sm[m]))
