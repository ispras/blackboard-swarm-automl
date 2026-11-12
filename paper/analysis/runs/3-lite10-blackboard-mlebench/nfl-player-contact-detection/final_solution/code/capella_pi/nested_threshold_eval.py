import pandas as pd, numpy as np
from sklearn.metrics import matthews_corrcoef
def best_mcc(y,p):
    ths=np.quantile(p,np.linspace(0.5,0.9995,300)); order=np.argsort(-p); ps=p[order]; ys=np.asarray(y)[order]
    cp=np.cumsum(ys,dtype=np.float64); cn=np.cumsum(1-ys,dtype=np.float64)
    Pos=int(ys.sum()); Neg=len(ys)-Pos
    idx=np.clip(np.searchsorted(-ps,-ths,side='left'),0,len(ys))
    tp=np.where(idx>0,cp[np.maximum(idx-1,0)],0.0); fp=np.where(idx>0,cn[np.maximum(idx-1,0)],0.0)
    fn=Pos-tp; tn=Neg-fp; den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+1e-12
    mcc=(tp*tn-fp*fn)/den; k=int(np.argmax(mcc)); return float(mcc[k]),float(ths[k])
def smooth_vals(p,gp,p1,p2,step,w):
    if w<=0: return np.asarray(p)
    d=pd.DataFrame({'p':np.asarray(p),'gp':gp,'p1':p1,'p2':p2,'s':step})
    d=d.sort_values(['gp','p1','p2','s']); g=d.groupby(['gp','p1','p2'],sort=False)['p']
    r=g.rolling(w,center=True,min_periods=1).mean(); r.index=r.index.droplevel([0,1,2])
    return r.reindex(d.index).sort_index().values
def logit(p,e=1e-6): p=np.clip(p,e,1-e); return np.log(p/(1-p))
st=pd.read_parquet('workspaces/capella_pi/oof_stack_mixv3.parquet')
en=pd.read_parquet('workspaces/canopus_pi/preds/ens_oof_probs_v4.parquet')[['contact_id','p_pp','p_gr']]
df=st.merge(en,on='contact_id',how='left')
sp=df.contact_id.str.split('_'); df['gp']=sp.str[0]+'_'+sp.str[1]; df['step']=sp.str[2].astype(int); df['p1']=sp.str[3]; df['p2']=sp.str[4]
ig=df.is_ground.values==1
df['ens']=np.where(ig,df.p_gr.values,df.p_pp.values)
df['hedge75']=0.75*logit(df.stack_oof.values)+0.25*logit(df.ens.values)
models={'stack_v3':'stack_oof','ens':'ens','hedge75':'hedge75'}
WINS=[0,1,2,3,5,7,9]
for mname,col in models.items():
    pred=np.zeros(len(df),dtype=np.int8)
    for task,igmask in [('pp',0),('ground',1)]:
        mask=(df.is_ground.values==igmask); idx=np.where(mask)[0]
        for f in range(5):
            tr_idx=idx[df.fold.values[idx]!=f]; te_idx=idx[df.fold.values[idx]==f]
            sub=df.iloc[tr_idx]
            best=(-1,0.5,0)
            for w in WINS:
                p=smooth_vals(df[col].values[tr_idx],sub.gp.values,sub.p1.values,sub.p2.values,sub.step.values,w)
                mcc,th=best_mcc(sub.contact.values,p)
                if mcc>best[0]: best=(mcc,th,w)
            _,th,w=best
            te=df.iloc[te_idx]
            p=smooth_vals(df[col].values[te_idx],te.gp.values,te.p1.values,te.p2.values,te.step.values,w)
            pred[te_idx]=(p>th).astype(np.int8)
    g=matthews_corrcoef(df.contact.values,pred)
    print(mname,'nested-threshold GLOBAL MCC',round(g,5),'pos rate',round(pred.mean(),5))
