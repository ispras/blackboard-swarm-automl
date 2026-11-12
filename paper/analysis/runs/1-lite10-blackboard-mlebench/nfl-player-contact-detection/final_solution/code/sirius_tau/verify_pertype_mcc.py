import pandas as pd, numpy as np, sys, time
from scipy.stats import rankdata
sys.path.insert(0,'workspaces/sirius_tau'); from lib_common import best_thr
t0=time.time()
P='workspaces/fomalhaut_omicron/'; Q='workspaces/procyon_theta/'
files={'feat5d2':P+'oof_pred_feat5_d2.parquet','feat5':P+'oof_pred_feat5.parquet','feat4':P+'oof_pred_feat4.parquet',
       'feat3d2':P+'oof_pred_feat3_d2.parquet','gruc':Q+'oof_pred_gruc.parquet','gru':Q+'oof_pred_gru.parquet',
       'gallr':Q+'oof_pred_gallr.parquet','l2':Q+'oof_pred_l2.parquet'}
base=pd.read_parquet(files['feat5d2'],columns=['contact_id']); cid=base.contact_id.values; n=len(cid)
lb=pd.read_csv('data/train_labels.csv',usecols=['contact_id','contact'])
y=pd.Series(lb.contact.values,index=lb.contact_id.values).reindex(cid).values.astype(np.int8)
isg=np.array([s.endswith('_G') for s in cid])
R={}
for k,f in files.items():
    o=pd.read_parquet(f)
    po=o.drop_duplicates('contact_id').set_index('contact_id')['prob'].reindex(cid).values
    R[k]=rankdata(po).astype(np.float32)/n
    print('loaded',k,round(time.time()-t0),flush=True)
def smooth(v,mask,w):
    # smooth within type using group keys from cid
    idx=np.where(mask)[0]; c=cid[idx]
    parts=pd.Series(c).str.rsplit('_',n=3,expand=True)
    df=pd.DataFrame({'gp':parts[0].values,'s':parts[1].values.astype(int),'p1':parts[2].values,'p2':parts[3].values,'v':v[idx]})
    order=np.argsort(df['s'].values,kind='stable'); d2=df.iloc[order]
    # use pandas sort
    df=df.sort_values(['gp','p1','p2','s'])
    sm=df.groupby(['gp','p1','p2'],sort=False)['v'].transform(lambda x:x.rolling(w,center=True,min_periods=1).mean()).values
    out=np.zeros(len(idx)); out[df.index.values]=sm; return out
# ground blend
gb=(R['feat5d2']+R['gruc']+R['gru']+R['feat5'])/4
gm=isg
bestg=(-1,None)
for w in [1,3,5,7]:
    s=smooth(gb,gm,w); _,m=best_thr(y[gm],s); print('ground w',w,'MCC',round(m,4),flush=True)
    if m>bestg[0]: bestg=(m,w)
# pair blend feat5x3 + gallr + l2 + feat5d2 + gruc
pb=(3*R['feat5']+R['gallr']+R['l2']+R['feat5d2']+R['gruc'])/7
pm=~isg; bestp=(-1,None)
for w in [1,3,5,7]:
    s=smooth(pb,pm,w); _,m=best_thr(y[pm],s); print('pair w',w,'MCC',round(m,4),flush=True)
    if m>bestp[0]: bestp=(m,w)
print('BEST ground',round(bestg[0],4),'w',bestg[1],'| pair',round(bestp[0],4),'w',bestp[1],flush=True)
