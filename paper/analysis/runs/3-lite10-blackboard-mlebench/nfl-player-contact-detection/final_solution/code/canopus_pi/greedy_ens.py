import os, time, json
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef
ROOT="/download_job_code/payload/sandbox"; P=f"{ROOT}/workspaces/canopus_pi/preds"; W=f"{ROOT}/workspaces/alkaid_psi"; C=f"{ROOT}/workspaces/capella_pi"
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)
def parse(df):
    p=df.contact_id.str.rsplit("_",n=3,expand=True); df=df.copy()
    df["gp"]=p[0].values; df["step"]=p[1].astype(int).values; df["p1"]=p[2].values; df["p2"]=p[3].values; return df
def lgt(p): p=np.clip(np.asarray(p,float),1e-6,1-1e-6); return np.log(p/(1-p))
def smooth_vals(p,df,keys,w):
    if w<=0: return np.asarray(p)
    d=df[keys+["step"]].copy(); d["p"]=np.asarray(p); d["_l"]=np.arange(len(p)); d=d.sort_values(keys+["step"])
    g=d.groupby(keys,sort=False)["p"]; r=g.rolling(w,center=True,min_periods=1).mean(); r.index=r.index.droplevel(list(range(len(keys))))
    return r.reindex(d.index).sort_index().values
def sweep(y,p,lo=0.3,hi=0.9999):
    o=np.argsort(-p,kind="stable"); ys=y[o].astype(np.int64); ps=p[o]
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); P=int(ys.sum()); N=len(y)-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp).astype(np.float64)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.maximum(den,1e-12),0.0)
    qlo,qhi=np.quantile(p,[lo,hi]); mcc=np.where((ps>=qlo)&(ps<=qhi),mcc,-1)
    i=int(np.argmax(mcc)); return float(mcc[i]),float(ps[i])
meta=pd.read_parquet(f"{ROOT}/workspaces/canopus_pi/features/all_features.parquet",columns=["contact_id","split","contact","is_ground"])
meta=parse(meta); tr=meta[meta.split==0].reset_index(drop=True); te=meta[meta.split==1].reset_index(drop=True)
tr["y"]=tr.contact.fillna(0).astype(np.int8)
WINS=[0,1,2,3,5,7]
def load(name,oof,test,ids_tr,ids_te):
    o=pd.read_parquet(oof).set_index("contact_id")
    col="pred" if "pred" in o.columns else [c for c in o.columns if c not in ("fold","contact")][0]
    t=pd.read_parquet(test).set_index("contact_id"); tc="pred" if "pred" in t.columns else [c for c in t.columns][0]
    a=o[col].reindex(ids_tr).values; b=t[tc].reindex(ids_te).values
    return a,b

def greedy(taskname, sources, igmask):
    sub=tr[tr.is_ground==igmask].reset_index(drop=True); tsub=te[te.is_ground==igmask].reset_index(drop=True)
    keys=["gp","p1","p2"]
    logits={}; tlogits={}
    for nm,of,tf in sources:
        a,b=load(nm,of,tf,sub.contact_id.values,tsub.contact_id.values)
        a=lgt(np.where(np.isfinite(a),a,np.nan)); b=lgt(np.where(np.isfinite(b),b,np.nan))
        # missing -> low logit (non-candidate)
        a=np.where(np.isfinite(a),a,-9.0); b=np.where(np.isfinite(b),b,-9.0)
        logits[nm]=a; tlogits[nm]=b
    y=sub.y.values
    cur=None; cur_mcc=-1; cur_w=0; chosen=[]
    for step in range(18):
        best=(-1,None,None)
        for nm,a in logits.items():
            p=a if cur is None else cur+a
            for w in WINS:
                ps=smooth_vals(p,sub,keys,w); m,th=sweep(y,ps)
                if m>best[0]: best=(m,nm,w,th,ps)
        m,nm,w,th,ps=best
        if m>cur_mcc+1e-6:
            cur = logits[nm] if cur is None else cur+logits[nm]
            cur_mcc=m; cur_w=w; cur_th=th; chosen.append(nm)
            log(taskname,"step",step,"add",nm,"mcc",round(m,6),"w",w,"th",round(th,4))
        else:
            break
    # target rate on OOF
    ps=smooth_vals(cur,sub,keys,cur_w); rate=(ps>=cur_th).mean()
    log(taskname,"FINAL",chosen,"mcc",round(cur_mcc,6),"rate",round(float(rate),6))
    # test
    tcur=None
    for nm in chosen: tcur=tlogits[nm] if tcur is None else tcur+tlogits[nm]
    tps=smooth_vals(tcur,tsub,keys,cur_w)
    rate_test=(tps>=cur_th).mean()
    return dict(chosen=chosen,w=cur_w,th=cur_th,mcc=cur_mcc,rate=float(rate),rate_test=float(rate_test),oof=ps,test=tps)

pp_src=[(t,f"{P}/oof_pp_{t}.parquet",f"{P}/test_pp_{t}.parquet") for t in ["v2","v3","v4","v5","gru_v1","gru_v2","gru_v3","gru_v4","xgb1"]]
pp_src += [("cap_lgbm",f"{C}/oof_lgbm.parquet",f"{C}/test_lgbm.parquet"),
           ("cnn_v2m",f"{W}/oof_cnn_cand_v2m.parquet",f"{W}/test_cnn_cand_v2m.parquet"),
           ("cnn_v2tta",f"{W}/oof_cnn_cand_v2tta.parquet",f"{W}/test_cnn_cand_v2tta.parquet")]
gr_src=[(t,f"{P}/oof_ground_{t}.parquet",f"{P}/test_ground_{t}.parquet") for t in ["v1","v2","v3","v4","gru_v1","gru_v2","gru_v3","gru_v4","gru_v5","xgb1"]]
gr_src += [("gcnn_v1m",f"{W}/oof_ground_cnn_v1m.parquet",f"{W}/test_ground_cnn_v1m.parquet"),
           ("gcnn_vB1m",f"{W}/oof_ground_cnn_vB1m.parquet",f"{W}/test_ground_cnn_vB1m.parquet"),
           ("gcnn_vC1m",f"{W}/oof_cnn_cand_v2m.parquet".replace("oof_cnn_cand_v2m","oof_ground_cnn_vC1m"),f"{W}/test_ground_cnn_vC1m.parquet"),
           ("cap_lgbm",f"{C}/oof_lgbm.parquet",f"{C}/test_lgbm.parquet")]
res_pp=greedy("PP",pp_src,False); res_gr=greedy("GR",gr_src,True)
# build submission with rate-matched thresholds per task
full=te[["contact_id","is_ground","gp","step"]].copy()
full["p_pp"]=np.nan; full["p_gr"]=np.nan
full.loc[te.is_ground.values==False,"p_pp"]=res_pp["test"]
full.loc[te.is_ground.values==True,"p_gr"]=res_gr["test"]
out=np.zeros(len(full),np.int8)
pp_idx=np.where(te.is_ground.values==False)[0]; gr_idx=np.where(te.is_ground.values==True)[0]
out[pp_idx]=(res_pp["test"]>np.quantile(res_pp["test"],1-res_pp["rate"])).astype(np.int8)
out[gr_idx]=(res_gr["test"]>np.quantile(res_gr["test"],1-res_gr["rate"])).astype(np.int8)
sub=pd.DataFrame({"contact_id":full.contact_id.values,"contact":out})
sub.to_csv(f"{P}/ens_submission_v5.csv",index=False)
oof=tr[["contact_id","contact","is_ground","gp","step"]].copy(); oof["p_pp"]=np.nan; oof["p_gr"]=np.nan
oof.loc[tr.is_ground.values==False,"p_pp"]=res_pp["oof"]; oof.loc[tr.is_ground.values==True,"p_gr"]=res_gr["oof"]
oof["pred"]=0
oof.loc[tr.is_ground.values==False,"pred"]=(res_pp["oof"]>res_pp["th"]).astype(np.int8)
oof.loc[tr.is_ground.values==True,"pred"]=(res_gr["oof"]>res_gr["th"]).astype(np.int8)
oof.to_parquet(f"{P}/ens_oof_probs_v5.parquet",index=False)
full["contact"]=out; full.to_parquet(f"{P}/ens_test_probs_v5.parquet",index=False)
# combined OOF mcc with independent thresholds
m=matthews_corrcoef(tr.contact.fillna(0).values.astype(int),oof.pred.values)
log("COMBINED OOF MCC",round(m,6),"test pos",round(out.mean(),6),"pp",round(out[pp_idx].mean(),6),"gr",round(out[gr_idx].mean(),6))
json.dump({"pp":{k:v for k,v in res_pp.items() if k not in ("oof","test")},
           "ground":{k:v for k,v in res_gr.items() if k not in ("oof","test")},"combined_oof":float(m),
           "test_pos":float(out.mean())},open(f"{P}/greedy_metrics.json","w"),indent=2)
log("saved")
