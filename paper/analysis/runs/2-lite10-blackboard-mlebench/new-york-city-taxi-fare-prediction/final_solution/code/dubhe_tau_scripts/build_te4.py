import numpy as np, json, time
ROOT="/download_job_code/payload/sandbox"; FEAT=f"{ROOT}/workspaces/dubhe_tau/features"
meta=json.load(open(f"{FEAT}/meta.json")); n=meta['n_valid']
X=np.load(f"{FEAT}/X_full.npy",mmap_mode='r'); y=np.load(f"{FEAT}/y_full.npy",mmap_mode='r'); ho=np.load(f"{FEAT}/holdout_full.npy",mmap_mode='r')
TE2=np.load(f"{FEAT}/TE_dec2.npy",mmap_mode='r')
yv=np.asarray(y[:n],np.float64); hm=np.asarray(ho[:n])
plon=np.asarray(X[:n,0],np.float64); plat=np.asarray(X[:n,1],np.float64); dlon=np.asarray(X[:n,2],np.float64); dlat=np.asarray(X[:n,3],np.float64)
year=np.asarray(X[:n,5]).astype(np.int64); hour=np.asarray(X[:n,8]).astype(np.int64); wd=np.asarray(X[:n,10]).astype(np.int64)
def cell(lat,lon,dec):
    m=10**dec; return (np.rint(lat*m).astype(np.int64))*(10**(dec+4))+(np.rint(lon*m).astype(np.int64))
pu2=cell(plat,plon,2); do2=cell(dlat,dlon,2); rt2=pu2*np.int64(1000003)+do2
yr=year-2009; hb=np.clip(hour//4,0,5)  # 6 buckets of 4h
print("keys",flush=True)
folds=np.full(n,-1,dtype=np.int8); tr=np.where(hm==0)[0]
rng=np.random.default_rng(555); folds[tr]=rng.integers(0,5,size=len(tr)).astype(np.int8)
prior=float(yv[tr].mean()); ALPHA=20.0
rt2_t=np.asarray(TE2[:n,4],dtype=np.float32)
out=np.empty((n,6),dtype=np.float32)
specs=[('rt_year', rt2*np.int64(16)+yr, 4), ('rt_hb', rt2*np.int64(16)+hb, 4)]
for j,(name,keys,slot) in enumerate(specs):
    uk,codes=np.unique(keys,return_inverse=True); K=len(uk)
    istr=folds>=0; cs=codes[istr]; fs=folds[istr]; ys=yv[istr]
    sum_tot=np.bincount(cs,weights=ys,minlength=K); cnt_tot=np.bincount(cs,minlength=K)
    combo=cs.astype(np.int64)*5+fs.astype(np.int64)
    sum_f=np.bincount(combo,weights=ys,minlength=K*5).reshape(K,5); cnt_f=np.bincount(combo,minlength=K*5).reshape(K,5)
    s=sum_tot[codes].copy(); c=cnt_tot[codes].astype(np.float64); f=folds; m=f>=0
    s[m]-=sum_f[codes[m],f[m]]; c[m]-=cnt_f[codes[m],f[m]]
    te=((s+prior*ALPHA)/(c+ALPHA)).astype(np.float32)
    back=c<1.0; te=np.where(back,rt2_t,te)
    out[:,3*j]=te; out[:,3*j+1]=np.log1p(c).astype(np.float32); out[:,3*j+2]=back.astype(np.float32)
    print(name,"K",K,"back",float(back.mean()),flush=True)
    Xt=np.load(f"{FEAT}/test_X.npy")
    tpu2=cell(Xt[:,1].astype(np.float64),Xt[:,0].astype(np.float64),2); tdo2=cell(Xt[:,3].astype(np.float64),Xt[:,2].astype(np.float64),2)
    trt2=tpu2*np.int64(1000003)+tdo2
    ty=(Xt[:,5].astype(np.int64)-2009); thb=np.clip(Xt[:,8].astype(np.int64)//4,0,5)
    tslot=ty if j==0 else thb
    tk=trt2*np.int64(16)+tslot
    pos=np.clip(np.searchsorted(uk,tk),0,K-1); found=uk[pos]==tk
    s2=np.where(found,sum_tot[pos],0.0); c2=np.where(found,cnt_tot[pos].astype(np.float64),0.0)
    teo=((s2+prior*ALPHA)/(c2+ALPHA)).astype(np.float32)
    tref=np.load(f"{FEAT}/test_TE_dec2.npy")[:,4]
    teo=np.where(found,teo,tref)
    te_out=np.column_stack([teo,np.log1p(c2).astype(np.float32),(~found).astype(np.float32)]).astype(np.float32)
    if j==0: te_all=te_out
    else: te_all=np.concatenate([te_all,te_out],axis=1)
np.save(f"{FEAT}/TE_extra4.npy",out); np.save(f"{FEAT}/test_TE_extra4.npy",te_all)
print("saved",out.shape,te_all.shape,flush=True)
