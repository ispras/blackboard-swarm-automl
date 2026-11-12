import numpy as np, pandas as pd, argparse, json, os
from scipy.optimize import nnls
ROOT="/download_job_code/payload/sandbox"; FEAT=f"{ROOT}/workspaces/dubhe_tau/features"
def rmse(p,y): return float(np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2)))
def load(specs):
    ref=pd.read_csv(f"{FEAT}/holdout_reference.csv").rename(columns={'fare_amount':'y'})
    T={}; names=[n for n,_,_ in specs]
    for n,h,t in specs:
        d=pd.read_csv(h); pc='pred' if 'pred' in d.columns else 'fare_amount'
        d=d.groupby('key',as_index=False)[pc].mean().rename(columns={pc:n})
        ref=ref.merge(d,on='key',how='inner')
        dt=pd.read_csv(t); tc='pred' if 'pred' in dt.columns else 'fare_amount'
        T[n]=dt.groupby('key',as_index=False)[tc].mean().rename(columns={tc:n})
    ref=ref.dropna(subset=names)
    return ref,names,T
def fit_nnls(P,y): return nnls(P,y)[0]
def fit_ridge(P,y,alpha=1.0):
    A=P.T@P + alpha*np.eye(P.shape[1]); return np.linalg.solve(A, P.T@y)
def cv(P,y,fitter,seed=0):
    rng=np.random.default_rng(seed); idx=rng.permutation(len(y)); folds=np.array_split(idx,5)
    pr=np.zeros(len(y))
    for k in range(5):
        te=folds[k]; tr=np.concatenate([folds[j] for j in range(5) if j!=k])
        w=fitter(P[tr],y[tr]); pr[te]=P[te]@w
    return rmse(pr,y)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preds',nargs='+',required=True)
    ap.add_argument('--tag',default='final'); ap.add_argument('--outdir',default=f"{ROOT}/workspaces/dubhe_tau/blend")
    ap.add_argument('--write_sub',default=''); ap.add_argument('--restrict',default=''); a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    specs=[]
    for s in a.preds:
        n,h,t=s.split(':'); specs.append((n,h,None if t=='-' else t))
    ref,names,T=load(specs)
    if a.restrict:
        rk=set(pd.read_csv(a.restrict)['key']); ref=ref[ref['key'].isin(rk)].reset_index(drop=True)
    y=ref['y'].values.astype(np.float64); P=ref[names].values.astype(np.float64)
    print("rows",len(y)); 
    for i,n in enumerate(names): print(f"  {n:16s} {rmse(P[:,i],y):.5f}")
    cvn=cv(P,y,fit_nnls); cvr=cv(P,y,lambda p,q: fit_ridge(p,q,1.0))
    print(f"CV nnls={cvn:.5f} ridge={cvr:.5f}")
    if cvr < cvn-1e-4: method='ridge'; w=fit_ridge(P,y,1.0)
    else: method='nnls'; w=fit_nnls(P,y)
    print("method",method,"weights",dict(zip(names,np.round(w,4))),"sum",w.sum(),"in-sample",rmse(P@w,y))
    json.dump({'names':names,'weights':w.tolist(),'method':method,'cv_nnls':cvn,'cv_ridge':cvr,
               'insample':rmse(P@w,y)},open(f"{a.outdir}/{a.tag}_weights.json",'w'),indent=2)
    # test
    test=pd.read_csv(f"{ROOT}/data/test.csv")[['key']]
    pred=np.zeros(len(test)); have=np.zeros(len(test))
    for i,n in enumerate(names):
        if n in T:
            m=test.merge(T[n],on='key',how='left'); v=m[n].values; ok=~np.isnan(v)
            pred[ok]+=v[ok]*w[i]; have[ok]+=w[i]
    pred=np.where(np.abs(have)>1e-9, pred/np.where(np.abs(have)>1e-9,have,1), np.nan)
    raw=pd.DataFrame({'key':test['key'],'fare_amount':pred})
    raw.to_csv(f"{a.outdir}/{a.tag}_test_raw.csv",index=False)
    print("raw nan",int(np.isnan(pred).sum()))
    if a.write_sub:
        import sys; sys.path.insert(0,f"{ROOT}/workspaces/dubhe_tau")
        import postprocess_final as DPP
        pp=DPP.apply_postprocess(pred, pd.read_csv(f"{ROOT}/data/test.csv"))
        pd.DataFrame({'key':test['key'],'fare_amount':pp}).to_csv(a.write_sub,index=False)
        print("wrote",a.write_sub,"stats",pd.Series(pp).describe().to_dict())
if __name__=='__main__': main()
