import pandas as pd, numpy as np, json, sys, time
from scipy.stats import rankdata
sys.path.insert(0,'workspaces/sirius_tau'); from lib_common import best_thr
OUT='workspaces/sirius_tau'; t0=time.time()
M=[
 ('fh_feat5','workspaces/fomalhaut_omicron/oof_pred_feat5.parquet','workspaces/fomalhaut_omicron/test_pred_feat5.parquet'),
 ('fh_feat5d2','workspaces/fomalhaut_omicron/oof_pred_feat5_d2.parquet','workspaces/fomalhaut_omicron/test_pred_feat5_d2.parquet'),
 ('fh_feat4','workspaces/fomalhaut_omicron/oof_pred_feat4.parquet','workspaces/fomalhaut_omicron/test_pred_feat4.parquet'),
 ('fh_feat3d2','workspaces/fomalhaut_omicron/oof_pred_feat3_d2.parquet','workspaces/fomalhaut_omicron/test_pred_feat3_d2.parquet'),
 ('pr_l2','workspaces/procyon_theta/oof_pred_l2.parquet','workspaces/procyon_theta/test_pred_l2.parquet'),
 ('pr_gallr','workspaces/procyon_theta/oof_pred_gallr.parquet','workspaces/procyon_theta/test_pred_gallr.parquet'),
 ('pr_gruc','workspaces/procyon_theta/oof_pred_gruc.parquet','workspaces/procyon_theta/test_pred_gruc.parquet'),
 ('pr_gru','workspaces/procyon_theta/oof_pred_gru.parquet','workspaces/procyon_theta/test_pred_gru.parquet'),
 ('pr_seqc','workspaces/procyon_theta/oof_pred_seqc.parquet','workspaces/procyon_theta/test_pred_seqc.parquet'),
 ('pr_cb','workspaces/procyon_theta/oof_pred_cb.parquet','workspaces/procyon_theta/test_pred_cb.parquet'),
 ('pr_base','workspaces/procyon_theta/oof_pred.parquet','workspaces/procyon_theta/test_pred.parquet'),
 ('sirius','workspaces/sirius_tau/oof_pred.parquet','workspaces/sirius_tau/test_pred.parquet'),
]
base=pd.read_parquet(M[0][1],columns=['contact_id']); cid=base.contact_id.values; n=len(cid)
lb=pd.read_csv('data/train_labels.csv',usecols=['contact_id','contact'])
y=pd.Series(lb.contact.values,index=lb.contact_id.values).reindex(cid).values.astype(np.int8)
isg=np.array([s.endswith('_G') for s in cid])
sub=pd.read_csv('data/sample_submission.csv'); tcid=sub.contact_id.values; isgt=np.array([s.endswith('_G') for s in tcid])
names=[];R=[];RT=[]
for nm,of,tt in M:
    o=pd.read_parquet(of); t=pd.read_parquet(tt)
    po=o.drop_duplicates('contact_id').set_index('contact_id')['prob'].reindex(cid).values
    pt=t.drop_duplicates('contact_id').set_index('contact_id')['prob'].reindex(tcid).values
    po=np.where(np.isnan(po),-99,po); pt=np.where(np.isnan(pt),-99,pt)
    R.append(rankdata(po).astype(np.float32)/n); RT.append(rankdata(pt).astype(np.float32)/len(tcid)); names.append(nm)
    print('loaded',nm,round(time.time()-t0),flush=True)
R=np.array(R); RT=np.array(RT)
def greedy(Rm,ym,sub_idx,cap=8):
    bm=-1;bi=0
    for i in range(len(names)):
        _,m=best_thr(ym[sub_idx],Rm[i][sub_idx])
        if m>bm: bm=m;bi=i
    sel=[bi];ws=Rm[bi].copy();best=bm;print(' start',names[bi],round(bm,4),flush=True)
    for it in range(cap):
        cb=best;ci=-1
        for i in range(len(names)):
            _,m=best_thr(ym[sub_idx],(ws+Rm[i])[sub_idx])
            if m>cb+1e-5: cb=m;ci=i
        if ci<0: break
        sel.append(ci);ws=ws+Rm[ci];best=cb;print(' add',names[ci],round(cb,4),flush=True)
    return sel,ws/len(sel),best
rng=np.random.RandomState(0)
print('ground',flush=True); sg,bsg,mg=greedy(R[:,isg],y[isg],np.arange(isg.sum()))
print('pair',flush=True); psub=rng.choice((~isg).sum(),900000,replace=False); sp,bsp,mp=greedy(R[:,~isg],y[~isg],psub)
print('ground sel',[names[i] for i in sg],round(mg,4)); print('pair sel',[names[i] for i in sp],round(mp,4))
comb=np.zeros(n); comb[isg]=bsg; comb[~isg]=bsp
def smooth(v,cids,w):
    parts=pd.Series(cids).str.rsplit('_',n=3,expand=True)
    df=pd.DataFrame({'gp':parts[0].values,'s':parts[1].values.astype(int),'p1':parts[2].values,'p2':parts[3].values,'v':v})
    df=df.sort_values(['gp','p1','p2','s'])
    sm=df.groupby(['gp','p1','p2'],sort=False)['v'].transform(lambda x:x.rolling(w,center=True,min_periods=1).mean()).values
    out=np.zeros(len(v)); out[df.index.values]=sm; return out
res={}
for w in [3,5]:
    thr,mcc=best_thr(y,smooth(comb,cid,w)); res[w]=(thr,mcc,(smooth(comb,cid,w)>=thr).mean()); print('w',w,'MCC',round(mcc,5),'q',round(res[w][2],4),flush=True)
w=3 if res[3][1]>=res[5][1] else 5; thr,mcc,q=res[w]
sgt=sum(RT[i] for i in sg)/len(sg); spt=sum(RT[i] for i in sp)/len(sp)
ct=np.zeros(len(tcid)); ct[isgt]=sgt[isgt]; ct[~isgt]=spt[~isgt]; s=smooth(ct,tcid,w)
thr_t=np.quantile(s,1-q); sub['contact']=(s>=thr_t).astype(int)
sub[['contact_id','contact']].to_csv(f'{OUT}/sub_verify2.csv',index=False)
print('BEST w',w,'MCC',round(mcc,5),'posrate',sub.contact.mean(),flush=True)
json.dump({'mcc':float(mcc),'w':w,'q':float(q),'g':[names[i] for i in sg],'p':[names[i] for i in sp]},open(f'{OUT}/verify2_meta.json','w'))
