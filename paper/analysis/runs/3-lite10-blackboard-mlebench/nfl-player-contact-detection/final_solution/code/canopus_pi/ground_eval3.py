import os, itertools, time, json
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef
ROOT="/download_job_code/payload/sandbox"; P=f"{ROOT}/workspaces/canopus_pi/preds"; W=f"{ROOT}/workspaces/alkaid_psi"
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)
def parse(df):
    p=df.contact_id.str.rsplit("_",n=3,expand=True); df=df.copy()
    df["gp"]=p[0].values; df["step"]=p[1].astype(int).values; df["p1"]=p[2].astype(np.int64).values
    df["p2n"]=pd.to_numeric(p[3],errors="coerce").fillna(-1).astype(np.int64).values; return df
def smooth(pred,df,keys,w):
    if w<=0: return pred
    d=df[keys+["step"]].copy(); d["p"]=pred; d["_l"]=np.arange(len(pred)); d=d.sort_values(keys+["step"])
    g=d.groupby(keys,sort=False)["p"]; arrs=[d["p"].values]
    for k in range(1,w+1): arrs += [g.shift(k).values,g.shift(-k).values]
    out=np.nanmean(np.vstack(arrs),axis=0); r=np.empty(len(pred),np.float32); r[d["_l"].values]=out; return r
def lgt(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def apply_cnn(p,cv,w):
    p=p.copy(); m=np.isfinite(cv)
    if w<=0 or m.sum()==0: return p
    z=(1-w)*lgt(p[m].astype(np.float64))+w*lgt(cv[m]); p[m]=(1/(1+np.exp(-z))).astype(np.float32); return p
def sweep(y,p,lo=0.4,hi=0.9998):
    o=np.argsort(-p,kind="stable"); ys=y[o].astype(np.int64); ps=p[o]
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); P=int(ys.sum()); N=len(y)-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp).astype(np.float64)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.maximum(den,1e-12),0.0)
    qlo,qhi=np.quantile(p,[lo,hi]); mcc=np.where((ps>=qlo)&(ps<=qhi),mcc,-1)
    i=int(np.argmax(mcc)); return float(mcc[i]),float(ps[i])
def conf(y,p,nq=500):
    qs=np.unique(np.quantile(p,np.linspace(0.3,0.9998,nq)))
    o=np.argsort(-p,kind="stable"); ys=y[o].astype(np.int64); ps=p[o]
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); idx=np.clip(np.searchsorted(-ps,-qs),0,len(ps)-1)
    return qs,tp[idx],fp[idx]

meta=pd.read_parquet(f"{ROOT}/workspaces/canopus_pi/features/all_features.parquet",columns=["contact_id","split","contact","is_ground"])
meta=parse(meta); tr=meta[meta.split==0].reset_index(drop=True); te=meta[meta.split==1].reset_index(drop=True)
tr["y"]=tr.contact.fillna(0).astype(np.int8); tr=tr[tr.y.notna()].reset_index(drop=True)

# PP
pp=tr[tr.is_ground==0].reset_index(drop=True)
ptags=["v2","v3","v4","v5","gru_v1","gru_v2","gru_v3","xgb1"]
for t in ptags:
    s=pd.read_parquet(f"{P}/oof_pp_{t}.parquet").set_index("contact_id")["pred"]; pp[t]=pp.contact_id.map(s).values
combos=[["gru_v1","v4"],["gru_v1","v4","xgb1"],["gru_v1","gru_v3","v4","xgb1"],
        ["gru_v1","gru_v2","gru_v3","v4","xgb1"],["gru_v3","v4"],["gru_v1","gru_v3","v4"],
        ["v3","v4","xgb1"],["v4"],["gru_v1","v2","v3","v4"]]
bestp=(-1,None)
for c in combos:
    if not all(x in ptags for x in c): continue
    p=np.mean([pp[x].values for x in c],axis=0)
    for sw in [0,1,2]:
        q=smooth(p,pp,["gp","p1","p2n"],sw); mm,th=sweep(pp.y.values,q)
        if mm>bestp[0]: bestp=(mm,{"combo":c,"sw":sw,"th":th,"p":q})
    log("PP",c,"best",bestp[0])
log("BEST PP",bestp[0],bestp[1]["combo"],bestp[1]["sw"],bestp[1]["th"])

# GROUND
gr=tr[tr.is_ground==1].reset_index(drop=True)
gtags=[t for t in ["v1","v2","gru_v1","gru_v2","gru_v3","gru_v4","gru_v5","xgb1"] if os.path.exists(f"{P}/oof_ground_{t}.parquet")]
for t in gtags:
    s=pd.read_parquet(f"{P}/oof_ground_{t}.parquet").set_index("contact_id")["pred"]; gr[t]=gr.contact_id.map(s).values
cnns={}
for name,of in [("v1m",f"{W}/oof_ground_cnn_v1m.parquet"),("vB1m",f"{W}/oof_ground_cnn_vB1m.parquet"),("vC1m",f"{W}/oof_ground_cnn_vC1m.parquet")]:
    if os.path.exists(of):
        s=pd.read_parquet(of).set_index("contact_id")["pred"]; cnns[name]=gr.contact_id.map(s).values
        log("cnn",name,"cov",np.isfinite(cnns[name]).mean())
bestg=(-1,None)
nw=len(gtags)
for wg in [w for w in itertools.product([0,.25,.5,.75,1.],repeat=nw) if abs(sum(w)-1)<1e-9]:
    base=np.zeros(len(gr),np.float64)
    for t,w in zip(gtags,wg): base+=w*gr[t].values
    for sw in [3,4,5,6]:
        q=smooth(base,gr,["gp","p1"],sw); mm,th=sweep(gr.y.values,q)
        if mm>bestg[0]: bestg=(mm,{"wg":wg,"sw":sw,"th":th,"p":q,"cnn":{}})
log("BEST GR base",bestg[0],bestg[1]["wg"],bestg[1]["sw"])
# greedy add cnns
for name,cv in cnns.items():
    cur=bestg[1]; improved=True
    while improved:
        improved=False; bx=bestg
        for w in [0.1,0.2,0.3,0.45]:
            p=apply_cnn(cur["p"],cv,w)
            for sw in [3,4,5,6]:
                q=smooth(p,gr,["gp","p1"],sw); mm,th=sweep(gr.y.values,q)
                if mm>bx[0] and mm>bestg[0]:
                    ne=dict(cur); ne["cnn"]=dict(cur["cnn"]); ne["cnn"][name]=w; ne["sw"]=sw; ne["th"]=th; ne["p"]=q
                    bx=(mm,ne)
        if bx[0]>bestg[0]+1e-6:
            bestg=bx; improved=True
    log("after",name,"mcc",bestg[0],bestg[1]["cnn"])
log("BEST GR",bestg[0],{k:v for k,v in bestg[1].items() if k!="p"})

qs_p,tp_p,fp_p=conf(pp.y.values,bestp[1]["p"]); qs_g,tp_g,fp_g=conf(gr.y.values,bestg[1]["p"])
Pt=pp.y.sum()+gr.y.sum(); N=len(pp)+len(gr)-Pt; gb=(-1,0,0)
for i,tpp in enumerate(qs_p):
    TP=tp_p[i]+tp_g; FP=fp_p[i]+fp_g; FN=Pt-TP; TN=N-FP
    den=np.sqrt((TP+FP).astype(np.float64)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc=np.where(den>0,(TP*TN-FP*FN)/np.maximum(den,1e-12),0); j=int(np.argmax(mcc))
    if mcc[j]>gb[0]: gb=(float(mcc[j]),float(tpp),float(qs_g[j]))
ind=matthews_corrcoef(np.r_[pp.y.values,gr.y.values],np.r_[bestp[1]["p"]>=bestp[1]["th"],bestg[1]["p"]>=bestg[1]["th"]])
log("GLOBAL 2D",gb,"indep",ind)

# ---- apply to test and save ens v4 ----
tpp_=te[te.is_ground==0].reset_index(drop=True); tgr_=te[te.is_ground==1].reset_index(drop=True)
for t in bestp[1]["combo"]:
    s=pd.read_parquet(f"{P}/test_pp_{t}.parquet").set_index("contact_id")["pred"]; tpp_[t]=tpp_.contact_id.map(s).values
ptest=np.mean([tpp_[t].values for t in bestp[1]["combo"]],axis=0)
ptest=smooth(ptest,tpp_,["gp","p1","p2n"],bestp[1]["sw"])
for t in gtags:
    s=pd.read_parquet(f"{P}/test_ground_{t}.parquet").set_index("contact_id")["pred"]; tgr_[t]=tgr_.contact_id.map(s).values
gtest=np.zeros(len(tgr_),np.float64)
for t,w in zip(gtags,bestg[1]["wg"]): gtest+=w*tgr_[t].values
for name,w in bestg[1]["cnn"].items():
    s=pd.read_parquet(f"{W}/test_ground_cnn_{name}.parquet").set_index("contact_id")["pred"]
    gtest=apply_cnn(gtest,tgr_.contact_id.map(s).values,w)
gtest=smooth(gtest,tgr_,["gp","p1"],bestg[1]["sw"])
full=te[["contact_id"]].copy()
full["p_pp"]=full.contact_id.map(dict(zip(tpp_.contact_id,ptest)))
full["p_gr"]=full.contact_id.map(dict(zip(tgr_.contact_id,gtest)))
full["contact"]=np.where(full.p_gr.notna(),full.p_gr>=gb[2],full.p_pp>=gb[1]).astype(np.int8)
log("test pos",full.contact.mean(),"pp",(full.p_pp>=gb[1]).mean(),"gr",(full.p_gr>=gb[2]).mean(),"nan",full.p_pp.isna().sum(),full.p_gr.isna().sum())
full.to_csv(f"{P}/ens_submission_v4.csv",index=False); full.to_parquet(f"{P}/ens_test_probs_v4.parquet",index=False)
oo=tr[["contact_id","contact","is_ground","gp","step"]].copy(); oo["p_pp"]=np.nan; oo["p_gr"]=np.nan
oo.loc[tr.is_ground.values==0,"p_pp"]=bestp[1]["p"]; oo.loc[tr.is_ground.values==1,"p_gr"]=bestg[1]["p"]
oo["pred"]=np.where(oo.p_gr.notna(),oo.p_gr>=gb[2],oo.p_pp>=gb[1]).astype(np.int8)
oo.to_parquet(f"{P}/ens_oof_probs_v4.parquet",index=False)
json.dump({"pp":{k:(list(v) if isinstance(v,(tuple,list)) else v) for k,v in bestp[1].items() if k!="p"},
           "pp_mcc":bestp[0],"ground_mcc":bestg[0],
           "ground":{k:(list(v) if isinstance(v,(tuple,list)) else (dict(v) if isinstance(v,dict) else v)) for k,v in bestg[1].items() if k!="p"},
           "global":list(gb),"global_indep":float(ind),"test_pos":float(full.contact.mean())},
          open(f"{P}/ens_metrics_v4.json","w"),indent=2)
log("saved")
