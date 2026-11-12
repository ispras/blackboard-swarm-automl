import pandas as pd, numpy as np, glob, os, sys, json, time
from scipy.stats import rankdata
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr

OUT='workspaces/sirius_tau'
t0=time.time()
MODELS=[
 ('fh_feat3','workspaces/fomalhaut_omicron/oof_pred_feat3.parquet','workspaces/fomalhaut_omicron/test_pred_feat3.parquet'),
 ('fh_feat3d2','workspaces/fomalhaut_omicron/oof_pred_feat3_d2.parquet','workspaces/fomalhaut_omicron/test_pred_feat3_d2.parquet'),
 ('fh_feat4','workspaces/fomalhaut_omicron/oof_pred_feat4.parquet','workspaces/fomalhaut_omicron/test_pred_feat4.parquet'),
 ('pr_base','workspaces/procyon_theta/oof_pred.parquet','workspaces/procyon_theta/test_pred.parquet'),
 ('pr_cb','workspaces/procyon_theta/oof_pred_cb.parquet','workspaces/procyon_theta/test_pred_cb.parquet'),
 ('pr_gallr','workspaces/procyon_theta/oof_pred_gallr.parquet','workspaces/procyon_theta/test_pred_gallr.parquet'),
 ('pr_l2','workspaces/procyon_theta/oof_pred_l2.parquet','workspaces/procyon_theta/test_pred_l2.parquet'),
 ('pr_seq','workspaces/procyon_theta/oof_pred_seq.parquet','workspaces/procyon_theta/test_pred_seq.parquet'),
 ('pr_seqc','workspaces/procyon_theta/oof_pred_seqc.parquet','workspaces/procyon_theta/test_pred_seqc.parquet'),
 ('pr_gruc','workspaces/procyon_theta/oof_pred_gruc.parquet','workspaces/procyon_theta/test_pred_gruc.parquet'),
 ('pr_gru','workspaces/procyon_theta/oof_pred_gru.parquet','workspaces/procyon_theta/test_pred_gru.parquet'),
 ('sirius','workspaces/sirius_tau/oof_pred.parquet','workspaces/sirius_tau/test_pred.parquet'),
]
base=pd.read_parquet(MODELS[0][1],columns=['contact_id'])
cid=base.contact_id.values; n=len(cid)
print('n',n,flush=True)
lb=pd.read_csv('data/train_labels.csv',usecols=['contact_id','contact'])
y=pd.Series(lb.contact.values,index=lb.contact_id.values).reindex(cid).values.astype(np.int8)
parts=pd.Series(cid).str.rsplit('_',n=3,expand=True)
gp=parts[0].values; step=parts[1].values; p1=parts[2].values; p2=parts[3].values
sub=pd.read_csv('data/sample_submission.csv'); tcid=sub.contact_id.values
tparts=pd.Series(tcid).str.rsplit('_',n=3,expand=True)
tgp=tparts[0].values; tstep=tparts[1].values; tp1=tparts[2].values; tp2=tparts[3].values

names=[]; R=[]; RT=[]
for nm,of,tt in MODELS:
    o=pd.read_parquet(of); t=pd.read_parquet(tt)
    o=pd.DataFrame({'contact_id':o.contact_id.values,'p':o.prob.values}).drop_duplicates('contact_id').set_index('contact_id')['p']
    t=pd.DataFrame({'contact_id':t.contact_id.values,'p':t.prob.values}).drop_duplicates('contact_id').set_index('contact_id')['p']
    po=o.reindex(cid).values; pt=t.reindex(tcid).values
    if np.isnan(po).any():
        po=np.where(np.isnan(po),np.nanmedian(po),po)
    if np.isnan(pt).any():
        pt=np.where(np.isnan(pt),np.nanmedian(pt),pt)
    R.append(rankdata(po).astype(np.float32)/n)
    RT.append(rankdata(pt).astype(np.float32)/len(tcid))
    names.append(nm); print('loaded',nm,'t',round(time.time()-t0),flush=True)
R=np.array(R); RT=np.array(RT)
np.save(f'{OUT}/blend_R.npy',R); np.save(f'{OUT}/blend_RT.npy',RT); np.save(f'{OUT}/blend_y.npy',y)
json.dump(names,open(f'{OUT}/blend_names.json','w'))

# greedy forward with integer weights on unsmoothed rank sum
def mcc_of(v):
    t,m=best_thr(y,v); return m,t
best_i=None; best_m=-1
for i in range(len(names)):
    m,_=mcc_of(R[i])
    print('single',names[i],round(m,4),flush=True)
    if m>best_m: best_m=m; best_i=i
sel=[best_i]; wsum=R[best_i].copy()
print('start',names[best_i],best_m,flush=True)
improved=True; it=0
while improved and it<25:
    improved=False; it+=1; cand_best=best_m; cand=-1
    for i in range(len(names)):
        m,_=mcc_of(wsum+R[i])
        if m>cand_best+1e-5:
            cand_best=m; cand=i
    if cand>=0:
        sel.append(cand); wsum=wsum+R[cand]; best_m=cand_best
        print('add',names[cand],'->',round(best_m,4),flush=True); improved=True
from collections import Counter
cnt=Counter([names[i] for i in sel]); print('WEIGHTS',dict(cnt),'unsmoothed MCC',round(best_m,4),flush=True)

# final: smooth blended rank (w5) and tune threshold
def smooth(v,keys_gp,keys_p1,keys_p2,keys_step,w=5):
    df=pd.DataFrame({'gp':keys_gp,'s':keys_step.astype(int),'p1':keys_p1,'p2':keys_p2,'v':v})
    df=df.sort_values(['gp','p1','p2','s'])
    sm=df.groupby(['gp','p1','p2'],sort=False)['v'].transform(lambda x:x.rolling(w,center=True,min_periods=1).mean()).values
    out=np.zeros(len(v)); out[df.index.values]=sm; return out
blend=wsum/len(sel)
bs=smooth(blend,gp,p1,p2,step,5)
thr,mcc=best_thr(y,bs); q=(bs>=thr).mean()
print('FINAL smoothed MCC',round(mcc,5),'thr',round(thr,5),'q',round(q,4),flush=True)
# test
wsum_t=sum(RT[i] for i in sel)/len(sel)
bts=smooth(wsum_t,tgp,tp1,tp2,tstep,5)
thr_t=np.quantile(bts,1-q)
sub['contact']=(bts>=thr_t).astype(int)
sub[['contact_id','contact']].to_csv(f'{OUT}/sub_myblend.csv',index=False)
np.save(f'{OUT}/blend_test.npy',bts)
json.dump({'mcc':float(mcc),'thr':float(thr),'q':float(q),'weights':dict(cnt)},open(f'{OUT}/blend_meta.json','w'))
print('saved sub_myblend posrate',sub.contact.mean(),'T',round(time.time()-t0),flush=True)
