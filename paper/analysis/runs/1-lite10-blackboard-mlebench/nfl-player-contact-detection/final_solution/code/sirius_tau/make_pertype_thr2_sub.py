import pandas as pd, numpy as np, json
from scipy.stats import rankdata
HERE='/download_job_code/payload/sandbox/workspaces/fomalhaut_omicron'
OUT='/download_job_code/payload/sandbox/workspaces/sirius_tau'
P=HERE+'/'; Q='/download_job_code/payload/sandbox/workspaces/procyon_theta/'
files={'feat5d2':P+'test_pred_feat5_d2.parquet','feat5':P+'test_pred_feat5.parquet',
       'gruc':Q+'test_pred_gruc.parquet','gru':Q+'test_pred_gru.parquet',
       'gallr':Q+'test_pred_gallr.parquet','l2':Q+'test_pred_l2.parquet'}
meta=json.load(open(OUT+'/win_thr.json')); qg=meta['qg']; qp=meta['qp']
ss=pd.read_csv('/download_job_code/payload/sandbox/data/sample_submission.csv')
cid=ss.contact_id.values; n=len(cid)
parts=pd.Series(cid).str.split('_')
gp=parts.str[0]+'_'+parts.str[1]; step=parts.str[2].astype(int); p1=parts.str[3]; p2=parts.str[4]
isg=(p2=='G').values
R={}
for k,f in files.items():
    o=pd.read_parquet(f).drop_duplicates('contact_id').set_index('contact_id')
    v=o['prob'].reindex(cid).values; v=np.nan_to_num(v,nan=0.0)
    R[k]=rankdata(v)/len(v)
wg={'feat5d2':1,'gruc':1,'gru':1,'feat5':1}
wp={'feat5':3,'gallr':1,'l2':1,'feat5d2':1,'gruc':1}
bg=sum(R[k]*c for k,c in wg.items())/sum(wg.values())
bp=sum(R[k]*c for k,c in wp.items())/sum(wp.values())
# build df, sort, group, smooth per type w5
df=pd.DataFrame({'contact_id':cid,'game_play':gp.values,'step':step.values,'p1':p1.values,'p2':p2.values,'bg':bg,'bp':bp,'isg':isg})
df=df.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
grp=df.groupby(['game_play','p1','p2'],sort=False).ngroup().values
def roll(vals,gg,w=5): return pd.Series(vals).groupby(gg).rolling(w,min_periods=1,center=True).mean().values
sc=np.empty(len(df))
gi=np.where(df.isg.values)[0]; pi=np.where(~df.isg.values)[0]
sc[gi]=roll(df.bg.values[gi],grp[gi],7); sc[pi]=roll(df.bp.values[pi],grp[pi],3)
df['sc']=sc
# per-type quantile thresholds
tg=np.quantile(df.sc.values[gi],1-qg); tp=np.quantile(df.sc.values[pi],1-qp)
df['pred']=np.where(df.isg.values,(df.sc.values>=tg),(df.sc.values>=tp)).astype(int)
out=df.set_index('contact_id')['pred'].reindex(ss.contact_id).reset_index(); out.columns=['contact_id','contact']
out.to_csv(OUT+'/sub_pertype_thr2.csv',index=False)
print('qg',qg,'qp',qp,'tg',tg,'tp',tp)
print('posrate',out.contact.mean(),'n',len(out),'ids ok',(out.contact_id.values==ss.contact_id.values).all())
print('ground posrate',out.contact.values[isg].mean(),'pair posrate',out.contact.values[~isg].mean())
