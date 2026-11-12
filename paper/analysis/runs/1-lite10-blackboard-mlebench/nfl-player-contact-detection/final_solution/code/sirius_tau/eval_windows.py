import pandas as pd, numpy as np, json, sys
from scipy.stats import rankdata
sys.path.insert(0,'workspaces/sirius_tau'); from lib_common import best_thr
OUT='workspaces/sirius_tau'
R=np.load(f'{OUT}/blend_R.npy'); RT=np.load(f'{OUT}/blend_RT.npy'); y=np.load(f'{OUT}/blend_y.npy')
names=json.load(open(f'{OUT}/blend_names.json')); ni={n:i for i,n in enumerate(names)}
base=pd.read_parquet('workspaces/procyon_theta/oof_pred.parquet',columns=['contact_id']); cid=base.contact_id.values
isg=np.array([s.endswith('_G') for s in cid])
sub=pd.read_csv('data/sample_submission.csv'); tcid=sub.contact_id.values
isgt=np.array([s.endswith('_G') for s in tcid])
G=['pr_gruc','pr_gallr','pr_gru','pr_seqc','fh_feat4']; P=['fh_feat3d2','pr_l2','pr_cb']
def comb_from(Rm):
    v=np.zeros(Rm.shape[1]); v[isg]=Rm[[ni[g] for g in G]].mean(0)[isg]; v[~isg]=Rm[[ni[p] for p in P]].mean(0)[~isg]; return v
comb=comb_from(R)
def smooth(v,cids,w):
    parts=pd.Series(cids).str.rsplit('_',n=3,expand=True)
    df=pd.DataFrame({'gp':parts[0].values,'s':parts[1].values.astype(int),'p1':parts[2].values,'p2':parts[3].values,'v':v})
    df=df.sort_values(['gp','p1','p2','s'])
    sm=df.groupby(['gp','p1','p2'],sort=False)['v'].transform(lambda x:x.rolling(w,center=True,min_periods=1).mean()).values
    out=np.zeros(len(v)); out[df.index.values]=sm; return out
best=(-1,None,None,None)
for w in [3,4,5,6,7,9]:
    s=smooth(comb,cid,w); thr,mcc=best_thr(y,s); q=(s>=thr).mean()
    print('w',w,'MCC',round(mcc,5),'thr',round(thr,5),'q',round(q,4),flush=True)
    if mcc>best[0]: best=(mcc,w,thr,q)
mcc,w,thr,q=best
combt=comb_from(RT); s=smooth(combt,tcid,w)
thr_t=np.quantile(s,1-q); sub['contact']=(s>=thr_t).astype(int)
sub[['contact_id','contact']].to_csv(f'{OUT}/sub_pertype_best.csv',index=False)
print('BEST w',w,'MCC',round(mcc,5),'q',round(q,4),'posrate',sub.contact.mean(),flush=True)
