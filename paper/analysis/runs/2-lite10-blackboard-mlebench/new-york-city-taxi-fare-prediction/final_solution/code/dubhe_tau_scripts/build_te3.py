import numpy as np, json, time
ROOT="/download_job_code/payload/sandbox"; FEAT=f"{ROOT}/workspaces/dubhe_tau/features"
meta=json.load(open(f"{FEAT}/meta.json")); n=meta['n_valid']
X=np.load(f"{FEAT}/X_full.npy",mmap_mode='r'); y=np.load(f"{FEAT}/y_full.npy",mmap_mode='r'); ho=np.load(f"{FEAT}/holdout_full.npy",mmap_mode='r')
TE2=np.load(f"{FEAT}/TE_dec2.npy",mmap_mode='r')
yv=np.asarray(y[:n],np.float64); hm=np.asarray(ho[:n])
plon=np.asarray(X[:n,0],np.float64); plat=np.asarray(X[:n,1],np.float64); dlon=np.asarray(X[:n,2],np.float64); dlat=np.asarray(X[:n,3],np.float64)
def cell(lat,lon,dec):
    m=10**dec; return (np.rint(lat*m).astype(np.int64))*(10**(dec+4))+(np.rint(lon*m).astype(np.int64))
t0=time.time()
pu3=cell(plat,plon,3); do3=cell(dlat,dlon,3); rt3=pu3*np.int64(1000003)+do3
print("keys",time.time()-t0,flush=True)
folds=np.full(n,-1,dtype=np.int8); tr=np.where(hm==0)[0]
rng=np.random.default_rng(999); folds[tr]=rng.integers(0,5,size=len(tr)).astype(np.int8)
prior=float(yv[tr].mean()); ALPHA=20.0; ALPHA2=50.0
out=np.empty((n,3),dtype=np.float32)
for j,(keys,name,al) in enumerate([(rt3,'rt3',ALPHA)]):
    uk,codes=np.unique(keys,return_inverse=True); K=len(uk)
    istr=folds>=0; cs=codes[istr]; fs=folds[istr]; ys=yv[istr]
    sum_tot=np.bincount(cs,weights=ys,minlength=K); cnt_tot=np.bincount(cs,minlength=K)
    combo=cs.astype(np.int64)*5+fs.astype(np.int64)
    sum_f=np.bincount(combo,weights=ys,minlength=K*5).reshape(K,5); cnt_f=np.bincount(combo,minlength=K*5).reshape(K,5)
    s=sum_tot[codes].copy(); c=cnt_tot[codes].astype(np.float64); f=folds; m=f>=0
    s[m]-=sum_f[codes[m],f[m]]; c[m]-=cnt_f[codes[m],f[m]]
    te=((s+prior*al)/(c+al)).astype(np.float32)
    rt2=np.asarray(TE2[:n,4],dtype=np.float32)
    back=c<1.0
    te=np.where(back,rt2,te)
    out[:,0]=te; out[:,1]=np.log1p(c).astype(np.float32); out[:,2]=back.astype(np.float32)
    print(name,"K",K,"backoff frac",float(back.mean()),time.time()-t0,flush=True)
    # test
    Xt=np.load(f"{FEAT}/test_X.npy")
    tpu3=cell(Xt[:,1].astype(np.float64),Xt[:,0].astype(np.float64),3); tdo3=cell(Xt[:,3].astype(np.float64),Xt[:,2].astype(np.float64),3)
    trt3=tpu3*np.int64(1000003)+tdo3
    pos=np.clip(np.searchsorted(uk,trt3),0,K-1); found=uk[pos]==trt3
    s2=np.where(found,sum_tot[pos],0.0); c2=np.where(found,cnt_tot[pos].astype(np.float64),0.0)
    te2o=((s2+prior*ALPHA2)/(c2+ALPHA2)).astype(np.float32)
    tTe2=np.load(f"{FEAT}/test_TE_dec2.npy")[:,4]
    fr=np.load(f"{FEAT}/test_X.npy")
    # rely on found flag as backoff indicator
    te2o=np.where(found,te2o,tTe2)
    teo=np.column_stack([te2o,np.log1p(c2).astype(np.float32),(~found).astype(np.float32)]).astype(np.float32)
np.save(f"{FEAT}/TE_extra3.npy",out); np.save(f"{FEAT}/test_TE_extra3.npy",teo)
print("saved",out.shape,time.time()-t0,flush=True)
