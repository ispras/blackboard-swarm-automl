import pandas as pd, numpy as np, json, sys, time
from scipy.stats import rankdata
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr
OUT='workspaces/sirius_tau'; t0=time.time()
R=np.load(f'{OUT}/blend_R.npy'); RT=np.load(f'{OUT}/blend_RT.npy'); y=np.load(f'{OUT}/blend_y.npy')
names=json.load(open(f'{OUT}/blend_names.json'))
base=pd.read_parquet('workspaces/procyon_theta/oof_pred.parquet',columns=['contact_id'])
cid=base.contact_id.values
isg=np.array([s.endswith('_G') for s in cid])
sub=pd.read_csv('data/sample_submission.csv'); tcid=sub.contact_id.values
isgt=np.array([s.endswith('_G') for s in tcid])
rng=np.random.RandomState(0)
sidx=rng.choice(len(y),700000,replace=False)

def greedy(Rm,ym,sub_idx,cap=8):
    bi=0; bm=-1
    for i in range(len(names)):
        _,m=best_thr(ym[sub_idx],Rm[i][sub_idx])
        if m>bm: bm=m; bi=i
    sel=[bi]; ws=Rm[bi].copy(); best=bm; print('  start',names[bi],round(bm,4),flush=True)
    for it in range(cap):
        cb=best; ci=-1
        for i in range(len(names)):
            _,m=best_thr(ym[sub_idx],(ws+Rm[i])[sub_idx])
            if m>cb+1e-5: cb=m; ci=i
        if ci<0: break
        sel.append(ci); ws=ws+Rm[ci]; best=cb; print('  add',names[ci],round(cb,4),flush=True)
    return sel,ws/len(sel),best

print('ground greedy',flush=True)
gsub=np.arange(isg.sum())
sg,bsg,mg=greedy(R[:,isg],y[isg],gsub)
print('ground',[names[i] for i in sg],round(mg,4),flush=True)
print('pair greedy',flush=True)
psub=rng.choice((~isg).sum(),700000,replace=False)
sp,bsp,mp=greedy(R[:,~isg],y[~isg],psub)
print('pair',[names[i] for i in sp],round(mp,4),flush=True)

comb=np.zeros(len(y)); comb[isg]=bsg; comb[~isg]=bsp
def smooth(v,cids,w=5):
    parts=pd.Series(cids).str.rsplit('_',n=3,expand=True)
    df=pd.DataFrame({'gp':parts[0].values,'s':parts[1].values.astype(int),'p1':parts[2].values,'p2':parts[3].values,'v':v})
    df=df.sort_values(['gp','p1','p2','s'])
    sm=df.groupby(['gp','p1','p2'],sort=False)['v'].transform(lambda x:x.rolling(w,center=True,min_periods=1).mean()).values
    out=np.zeros(len(v)); out[df.index.values]=sm; return out
combs=smooth(comb,cid,5)
thr,mcc=best_thr(y,combs); q=(combs>=thr).mean()
print('COMBINED raw',round(best_thr(y,comb)[1],5),'SMOOTH',round(mcc,5),'q',round(q,4),flush=True)

sgt=sum(RT[i] for i in sg)/len(sg); spt=sum(RT[i] for i in sp)/len(sp)
combt0=np.zeros(len(tcid)); combt0[isgt]=sgt[isgt]; combt0[~isgt]=spt[~isgt]
combt=smooth(combt0,tcid,5)
thr_t=np.quantile(combt,1-q)
sub['contact']=(combt>=thr_t).astype(int)
sub[['contact_id','contact']].to_csv(f'{OUT}/sub_pertype3.csv',index=False)
json.dump({'mcc':float(mcc),'q':float(q),'g':[names[i] for i in sg],'p':[names[i] for i in sp]},open(f'{OUT}/pertype3_meta.json','w'))
print('saved sub_pertype posrate',sub.contact.mean(),'T',round(time.time()-t0),flush=True)
