import argparse, json, os, time
import numpy as np, pandas as pd, xgboost as xgb
ROOT="/download_job_code/payload/sandbox"; FEAT=f"{ROOT}/workspaces/dubhe_tau/features"; OUT=f"{ROOT}/workspaces/dubhe_tau/models"
os.makedirs(OUT,exist_ok=True)
C30=np.cos(np.pi/6); S30=np.sin(np.pi/6)
def augment(X):
    plon=X[:,0];plat=X[:,1];dlon=X[:,2];dlat=X[:,3];hav=X[:,12];manh=X[:,13];bearing=X[:,14]
    hour=X[:,8];weekday=X[:,10];dpc=X[:,19];ddc=X[:,20]
    pj=X[:,21];pl=X[:,23];pe=X[:,25];dj=X[:,22];dl=X[:,24];de=X[:,26]
    b=np.radians(bearing)
    extra=np.column_stack([plon*C30-plat*S30,plon*S30+plat*C30,dlon*C30-dlat*S30,dlon*S30+dlat*C30,
        np.sin(hour*np.pi/12),np.cos(hour*np.pi/12),np.sin(weekday*2*np.pi/7),np.cos(weekday*2*np.pi/7),
        (weekday>=5).astype(np.float32),np.log1p(np.maximum(hav,0)),np.log1p(np.maximum(manh,0)),
        (hav<0.1).astype(np.float32),np.sin(b),np.cos(b),
        (np.minimum(pj,dj)<2).astype(np.float32),(np.minimum(pl,dl)<2).astype(np.float32),(np.minimum(pe,de)<2).astype(np.float32),
        ddc-dpc]).astype(np.float32)
    return np.concatenate([X,extra],axis=1)
p=argparse.ArgumentParser()
p.add_argument("--nrows",type=int,default=0); p.add_argument("--batch",type=int,default=4000000)
p.add_argument("--depth",type=int,default=13); p.add_argument("--lr",type=float,default=0.045)
p.add_argument("--rounds",type=int,default=6000); p.add_argument("--es",type=int,default=120)
p.add_argument("--seed",type=int,default=1); p.add_argument("--min_child_weight",type=float,default=30)
p.add_argument("--subsample",type=float,default=0.85); p.add_argument("--colsample",type=float,default=0.85)
p.add_argument("--max_bin",type=int,default=512); p.add_argument("--tag",type=str,default="xgb_te")
p.add_argument("--grow_policy",type=str,default="depthwise"); p.add_argument("--max_leaves",type=int,default=0)
p.add_argument("--device",type=str,default="cuda")
p.add_argument("--te2",type=int,default=0)
p.add_argument("--te3",type=int,default=0)
p.add_argument("--te4",type=int,default=0)
p.add_argument("--teres",type=int,default=0)
a=p.parse_args()
meta=json.load(open(f"{FEAT}/meta.json")); n=meta['n_valid']
X=np.load(f"{FEAT}/X_full.npy",mmap_mode='r'); y=np.load(f"{FEAT}/y_full.npy",mmap_mode='r')
ho=np.load(f"{FEAT}/holdout_full.npy",mmap_mode='r'); TE=np.load(f"{FEAT}/TE_dec2.npy",mmap_mode='r')
TE2=np.load(f"{FEAT}/TE_extra.npy",mmap_mode='r') if a.te2 else None
TE3=np.load(f"{FEAT}/TE_extra3.npy",mmap_mode='r') if a.te3 else None
TE4=np.load(f"{FEAT}/TE_extra4.npy",mmap_mode='r') if a.te4 else None
TR=np.load(f"{FEAT}/TE_res.npy",mmap_mode='r') if a.teres else None
hold_idx=np.where(np.asarray(ho[:n])==1)[0]; tr=np.where(np.asarray(ho[:n])==0)[0]
rng=np.random.default_rng(a.seed)
if a.nrows and a.nrows<len(tr): tr=rng.choice(tr,size=a.nrows,replace=False)
tr=tr.astype(np.int64)
print("train",len(tr),"hold",len(hold_idx),flush=True)
t0=time.time(); B=a.batch; order=rng.permutation(len(tr)); nb=(len(order)+B-1)//B
class It(xgb.DataIter):
    def __init__(self): super().__init__(cache_prefix=None); self.i=0
    def next(self,input_data):
        if self.i>=nb: return False
        ii=tr[order[self.i*B:(self.i+1)*B]]
        Xb=augment(np.asarray(X[ii],dtype=np.float32))
        Tb=np.asarray(TE[ii],dtype=np.float32)
        if a.te2: Tb=np.concatenate([Tb,np.asarray(TE2[ii],dtype=np.float32)],axis=1)
        if a.te3: Tb=np.concatenate([Tb,np.asarray(TE3[ii],dtype=np.float32)],axis=1)
        if a.te4: Tb=np.concatenate([Tb,np.asarray(TE4[ii],dtype=np.float32)],axis=1)
        if a.teres: Tb=np.concatenate([Tb,np.asarray(TR[ii],dtype=np.float32)],axis=1)
        input_data(data=np.concatenate([Xb,Tb],axis=1),label=np.asarray(y[ii],dtype=np.float32))
        self.i+=1; return True
    def reset(self): self.i=0
dtr=xgb.QuantileDMatrix(It(),max_bin=a.max_bin); print("dtrain built",time.time()-t0,flush=True)
Xva=np.concatenate([augment(np.asarray(X[hold_idx],dtype=np.float32)),np.asarray(TE[hold_idx],dtype=np.float32)],axis=1)
if a.te2: Xva=np.concatenate([Xva,np.asarray(TE2[hold_idx],dtype=np.float32)],axis=1)
if a.te3: Xva=np.concatenate([Xva,np.asarray(TE3[hold_idx],dtype=np.float32)],axis=1)
if a.te4: Xva=np.concatenate([Xva,np.asarray(TE4[hold_idx],dtype=np.float32)],axis=1)
if a.teres: Xva=np.concatenate([Xva,np.asarray(TR[hold_idx],dtype=np.float32)],axis=1)
yva=np.asarray(y[hold_idx],dtype=np.float32)
dva=xgb.QuantileDMatrix(Xva,label=yva,ref=dtr,max_bin=a.max_bin); print("dval built",Xva.shape,time.time()-t0,flush=True)
params=dict(objective="reg:squarederror",eval_metric="rmse",tree_method="hist",device=a.device,
    max_depth=a.depth,learning_rate=a.lr,subsample=a.subsample,colsample_bytree=a.colsample,
    min_child_weight=a.min_child_weight,reg_lambda=1.0,max_bin=a.max_bin,seed=a.seed,grow_policy=a.grow_policy)
if a.grow_policy=="lossguide" and a.max_leaves>0: params["max_leaves"]=a.max_leaves
bst=xgb.train(params,dtr,num_boost_round=a.rounds,evals=[(dva,"val")],early_stopping_rounds=a.es,verbose_eval=50)
best=bst.best_iteration; pv=bst.predict(dva,iteration_range=(0,best+1))
rmse=float(np.sqrt(np.mean((pv-yva)**2))); print(f"BEST_ITER {best} HOLDOUT_RMSE {rmse:.6f}",flush=True)
bst.save_model(f"{OUT}/{a.tag}.json"); np.save(f"{OUT}/{a.tag}_valpred.npy",pv.astype(np.float32)); np.save(f"{OUT}/{a.tag}_validx.npy",hold_idx.astype(np.int64))
Xte=np.concatenate([augment(np.load(f"{FEAT}/test_X.npy")),np.load(f"{FEAT}/test_TE_dec2.npy")],axis=1)
if a.te2: Xte=np.concatenate([Xte,np.load(f"{FEAT}/test_TE_extra.npy")],axis=1)
if a.te3: Xte=np.concatenate([Xte,np.load(f"{FEAT}/test_TE_extra3.npy")],axis=1)
if a.te4: Xte=np.concatenate([Xte,np.load(f"{FEAT}/test_TE_extra4.npy")],axis=1)
if a.teres: Xte=np.concatenate([Xte,np.load(f"{FEAT}/test_TE_res.npy")],axis=1)
pt=bst.predict(xgb.DMatrix(Xte),iteration_range=(0,best+1))
pd.DataFrame({"key":pd.read_csv(f"{FEAT}/test_keys.csv")['key'],"fare_amount":pt}).to_csv(f"{OUT}/{a.tag}_test.csv",index=False)
json.dump({"tag":a.tag,"holdout_rmse":rmse,"best_iter":int(best),"nrows":int(len(tr)),"args":vars(a)},open(f"{OUT}/{a.tag}_result.json","w"),indent=2)
print("SAVED",flush=True)
