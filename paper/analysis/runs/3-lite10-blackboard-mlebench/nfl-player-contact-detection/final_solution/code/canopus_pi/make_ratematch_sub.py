import pandas as pd, numpy as np
ROOT="/download_job_code/payload/sandbox"
def smooth_vals(p,gp,p1,p2,step,w):
    d=pd.DataFrame({'p':np.asarray(p),'gp':gp,'p1':p1,'p2':p2,'s':step})
    d=d.sort_values(['gp','p1','p2','s']); g=d.groupby(['gp','p1','p2'],sort=False)['p']
    r=g.rolling(w,center=True,min_periods=1).mean(); r.index=r.index.droplevel([0,1,2])
    return r.reindex(d.index).sort_index().values
ss=pd.read_csv(f"{ROOT}/data/sample_submission.csv")
mix=pd.read_parquet(f"{ROOT}/workspaces/capella_pi/test_stack_mix.parquet")
m=mix.set_index("contact_id")["stack_pred"].reindex(ss.contact_id.values)
print("missing",m.isna().sum())
sp=ss.contact_id.str.split("_")
gp=sp.str[0]+"_"+sp.str[1]; step=sp.str[2].astype(int); p1=sp.str[3]; p2=sp.str[4]
ig=(p2=="G").values
p=m.values.astype(np.float64)
# smooth per task with w=3 (same as OOF procedure)
out=np.zeros(len(ss),dtype=np.int8)
TARGET={"pp":0.012512,"ground":0.047549}
for task,mask in [("pp",~ig),("ground",ig)]:
    idx=np.where(mask)[0]
    ps=smooth_vals(p[idx],gp.values[idx],p1.values[idx],p2.values[idx],step.values[idx],3)
    th=np.quantile(ps,1-TARGET[task])
    out[idx]=(ps>th).astype(np.int8)
    print(task,"n",len(idx),"th",round(float(th),5),"pos_rate",round(float(out[idx].mean()),6),"target",TARGET[task])
sub=pd.DataFrame({"contact_id":ss.contact_id.values,"contact":out})
sub.to_csv(f"{ROOT}/workspaces/canopus_pi/preds/ens_submission_ratematch.csv",index=False)
print("total pos rate",out.mean(),"rows",len(sub))
