import pandas as pd, numpy as np, json
from scipy.stats import rankdata
from lib_common import best_thr
HERE='/download_job_code/payload/sandbox/workspaces/fomalhaut_omicron'
OUT='/download_job_code/payload/sandbox/workspaces/sirius_tau'
P=HERE+'/'; Q='/download_job_code/payload/sandbox/workspaces/procyon_theta/'
files={'feat5d2':P+'oof_pred_feat5_d2.parquet','feat5':P+'oof_pred_feat5.parquet',
       'gruc':Q+'oof_pred_gruc.parquet','gru':Q+'oof_pred_gru.parquet',
       'gallr':Q+'oof_pred_gallr.parquet','l2':Q+'oof_pred_l2.parquet'}
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
    v=o['prob'].reindex(ref.contact_id.values).values; v=np.nan_to_num(v,nan=np.nanmedian(v))
    R[k]=rankdata(v)/len(v)
wg={'feat5d2':1,'gruc':1,'gru':1,'feat5':1}
wp={'feat5':3,'gallr':1,'l2':1,'feat5d2':1,'gruc':1}
bg=sum(R[n]*c for n,c in wg.items())/sum(wg.values())
bp=sum(R[n]*c for n,c in wp.items())/sum(wp.values())
gi=np.where(isg)[0]; pi=np.where(~isg)[0]
def roll(vals,g,w): return pd.Series(vals).groupby(g).rolling(w,min_periods=1,center=True).mean().values
def cum(sc,m):
    s=sc[m]; ys=y[m]; o=np.argsort(-s); ys=ys[o]
    return np.concatenate([[0],np.cumsum(ys)]), np.concatenate([[0],np.cumsum(1-ys)]), s[o]
Kg=set(R); Kp=set(R)
def eval_combo(wgv,wpv,grid=2500):
    sc=np.empty(len(y)); sc[gi]=roll(bg[gi],grp[gi],wgv); sc[pi]=roll(bp[pi],grp[pi],wpv)
    tpg,fpg,svg=cum(sc,isg); tpp,fpp,svp=cum(sc,~isg)
    Pg=tpg[-1]; Pp=tpp[-1]; Ng=len(svg)-Pg; Np=len(svp)-Pp; Pat=Pg+Pp; Nat=Ng+Np
    kg=np.linspace(0,len(svg),grid).astype(int); kp=np.linspace(0,len(svp),grid).astype(int)
    best=(-1,0,0)
    tpgv=tpg[kg]; fpgv=fpg[kg]
    for a in range(len(kg)):
        TP=tpgv[a]+tpp[kp]; FP=fpgv[a]+fpp[kp]; FN=Pat-TP; TN=Nat-FP
        den=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        m=np.where(den>0,(TP*TN-FP*FN)/np.where(den>0,den,1),0)
        j=int(np.argmax(m))
        if m[j]>best[0]: best=(float(m[j]),int(kg[a]),int(kp[j]))
    m,i,j=best
    return m, i/len(svg), j/len(svp), float(best_thr(y.astype(int),sc)[1])
res=[]
for a,b in [(5,5),(7,3),(9,3),(7,5),(5,3),(3,3)]:
        m,qg,qp,msingle=eval_combo(a,b)
        res.append((m,a,b,qg,qp,msingle)); print('wg%d wp%d pertype %.5f single %.5f qg %.4f qp %.4f'%(a,b,m,msingle,qg,qp),flush=True)
res.sort(reverse=True)
m,a,b,qg,qp,ms=res[0]
print('BEST pertype %.5f wg%d wp%d qg %.5f qp %.5f (single-on-best %.5f)'%(m,a,b,qg,qp,ms),flush=True)
res.sort(reverse=True)
json.dump({'best':float(m),'wg':a,'wp':b,'qg':float(qg),'qp':float(qp),'all':[(float(x[0]),x[1],x[2]) for x in res]},open(OUT+'/win_thr.json','w'))
