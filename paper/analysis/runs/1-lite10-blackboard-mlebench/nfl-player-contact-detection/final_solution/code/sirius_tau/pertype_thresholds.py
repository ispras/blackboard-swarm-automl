import pandas as pd, numpy as np, os, sys, json
from scipy.stats import rankdata
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr
HERE='/download_job_code/payload/sandbox/workspaces/fomalhaut_omicron'
P=HERE+'/'; Q='/download_job_code/payload/sandbox/workspaces/procyon_theta/'
files={'feat5d2':P+'oof_pred_feat5_d2.parquet','feat5':P+'oof_pred_feat5.parquet',
       'gruc':Q+'oof_pred_gruc.parquet','gru':Q+'oof_pred_gru.parquet',
       'gallr':Q+'oof_pred_gallr.parquet','l2':Q+'oof_pred_l2.parquet'}
tfiles={k:v.replace('oof_pred','test_pred') for k,v in files.items()}

lab=pd.read_csv('/download_job_code/payload/sandbox/data/train_labels.csv',usecols=['contact_id','contact'])
ref=pd.read_parquet(files['feat5'],columns=['contact_id']).merge(lab,on='contact_id',how='left')
p=ref.contact_id.str.split('_')
ref['game_play']=p.str[0]+'_'+p.str[1]; ref['step']=p.str[2].astype(int); ref['p1']=p.str[3]; ref['p2']=p.str[4]
ref=ref.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
grp=ref.groupby(['game_play','p1','p2'],sort=False).ngroup().values
y=ref.contact.values.astype(np.float64); isg=(ref.p2=='G').values
R={}
for k,f in files.items():
    o=pd.read_parquet(f).drop_duplicates('contact_id').set_index('contact_id')
    v=o['prob'].reindex(ref.contact_id.values).values
    v=np.nan_to_num(v,nan=np.nanmedian(v)); R[k]=rankdata(v)/len(v)
def roll(vals,grp,w): return pd.Series(vals).groupby(grp).rolling(w,min_periods=1,center=True).mean().values
def combine(bg,bp,wg,wp):
    out=np.empty(len(bg)); gi=np.where(isg)[0]; pi=np.where(~isg)[0]
    out[gi]=roll(bg[gi],grp[gi],wg); out[pi]=roll(bp[pi],grp[pi],wp); return out
# final blend (from ens_pertype2 result)
wg={'feat5d2':1,'gruc':1,'gru':1,'feat5':1}
wp={'feat5':3,'gallr':1,'l2':1,'feat5d2':1,'gruc':1}
bg=sum(R[n]*c for n,c in wg.items())/sum(wg.values())
bp=sum(R[n]*c for n,c in wp.items())/sum(wp.values())
sc=combine(bg,bp,5,5)
m1,t1=best_thr(y.astype(int),sc); q1=float((sc>=t1).mean())
print('single global thr MCC %.5f thr %.5f q %.5f'%(m1,t1,q1),flush=True)

# per-type threshold 2D optimization via cumulative counts
def cum(scores,m):
    s=scores[m]; ys=y[m]
    o=np.argsort(-s); ys=ys[o]
    tp=np.concatenate([[0],np.cumsum(ys)]); fp=np.concatenate([[0],np.cumsum(1-ys)])
    # threshold value after k positives (k=0..len)
    sv=s[o]
    return tp,fp,sv
tpg,fpg,svg=cum(sc,isg); tpp,fpp,svp=cum(sc,~isg)
Pg=tpg[-1]; Pp=tpp[-1]; Ng=len(svg)-Pg; Np=len(svp)-Pp; Pat=Pg+Pp; Nat=Ng+Np
# grid indices
kg=np.linspace(0,len(svg),400).astype(int); kp=np.linspace(0,len(svp),400).astype(int)
best=(-1,None,None)
for ig in kg:
    TPg=tpg[ig]; FPg=fpg[ig]
    for jp in kp:
        TP=TPg+tpp[jp]; FP=FPg+fpp[jp]; FN=Pat-TP; TN=Nat-FP
        den=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if den>0:
            m=(TP*TN-FP*FN)/den
            if m>best[0]: best=(m,ig,jp)
m2,ig,jp=best
qg=float(ig/len(svg)); qp=float(jp/len(svp))
# threshold values (score after ig in ground)
tg=float(svg[ig]) if ig<len(svg) else -1
tpv=float(svp[jp]) if jp<len(svp) else -1
print('per-type thr MCC %.5f qg %.5f qp %.5f tg %.5f tp %.5f'%(m2,qg,qp,tg,tpv),flush=True)
print('delta vs single %.5f'%(m2-m1),flush=True)
json.dump({'single':float(m1),'pertype':float(m2),'qg':qg,'qp':qp,'q1':q1},open('/download_job_code/payload/sandbox/workspaces/sirius_tau/pertype_thr.json','w'))
