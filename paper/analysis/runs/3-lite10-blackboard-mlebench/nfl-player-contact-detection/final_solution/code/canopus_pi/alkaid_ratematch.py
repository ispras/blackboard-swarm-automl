import pandas as pd, numpy as np
from sklearn.metrics import matthews_corrcoef
ROOT="/download_job_code/payload/sandbox"
def best_mcc(y,p):
    ths=np.quantile(p,np.linspace(0.5,0.9995,500)); order=np.argsort(-p); ps=p[order]; ys=np.asarray(y)[order]
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
o=pd.read_parquet(f"{ROOT}/workspaces/alkaid_psi/oof_blend3.parquet")
sp=o.contact_id.str.split("_"); o['gp']=sp.str[0]+"_"+sp.str[1]; o['step']=sp.str[2].astype(int); o['p1']=sp.str[3]; o['p2']=sp.str[4]
WS={"pp":1,"ground":2}; rates={}
print("OOF optimal with fixed ws")
for task,ig,w in [("pp",False,1),("ground",True,2)]:
    sub=o[o.is_ground==ig]
    p=smooth_vals(sub.score.values,sub.gp.values,sub.p1.values,sub.p2.values,sub.step.values,w)
    mcc,th=best_mcc(sub.contact.values,p); rate=(p>=th).mean(); rates[task]=(th,rate)
    print(task,"n",len(sub),"mcc",round(mcc,6),"th",round(th,5),"rate",round(rate,6))
# nested fixed vs rate-matched
pf=np.zeros(len(o),np.int8); pq=np.zeros(len(o),np.int8)
for task,ig,w in [("pp",False,1),("ground",True,2)]:
    idx=np.where((o.is_ground==ig).values)[0]
    for f in range(5):
        tr=idx[o.fold.values[idx]!=f]; te=idx[o.fold.values[idx]==f]
        sub=o.iloc[tr]; p=smooth_vals(sub.score.values,sub.gp.values,sub.p1.values,sub.p2.values,sub.step.values,w)
        _,th=best_mcc(sub.contact.values,p); rate=(p>=th).mean()
        t=o.iloc[te]; pt=smooth_vals(t.score.values,t.gp.values,t.p1.values,t.p2.values,t.step.values,w)
        pf[te]=(pt>=th); pq[te]=(pt>=np.quantile(pt,1-rate))
print("nested fixed",round(matthews_corrcoef(o.contact.values,pf),6))
print("nested rate-matched",round(matthews_corrcoef(o.contact.values,pq),6))
# test submission
t=pd.read_parquet(f"{ROOT}/workspaces/alkaid_psi/test_blend3_probs.parquet")
ss=pd.read_csv(f"{ROOT}/data/sample_submission.csv")
t=t.set_index("contact_id").reindex(ss.contact_id.values)
sp=ss.contact_id.str.split("_"); gp=sp.str[0]+"_"+sp.str[1]; step=sp.str[2].astype(int); p1=sp.str[3]; p2=sp.str[4]
ig=t.is_ground.values
out=np.zeros(len(ss),np.int8)
for task,igv,w in [("pp",False,1),("ground",True,2)]:
    idx=np.where(ig==igv)[0]; p=smooth_vals(t.score.values[idx],gp.values[idx],p1.values[idx],p2.values[idx],step.values[idx],w)
    rate=rates[task][1]; th=np.quantile(p,1-rate); out[idx]=(p>th).astype(np.int8)
    print(task,"test th",round(float(th),5),"rate",round(float(out[idx].mean()),6),"target",round(rate,6))
sub=pd.DataFrame({"contact_id":ss.contact_id.values,"contact":out})
sub.to_csv(f"{ROOT}/workspaces/canopus_pi/preds/alkaid_blend3_ratematch.csv",index=False)
print("total pos",out.mean())
