import numpy as np, json, time
ROOT="/download_job_code/payload/sandbox"; FEAT=f"{ROOT}/workspaces/dubhe_tau/features"
meta=json.load(open(f"{FEAT}/meta.json")); n=meta['n_valid']
X=np.load(f"{FEAT}/X_full.npy",mmap_mode='r'); y=np.load(f"{FEAT}/y_full.npy",mmap_mode='r'); ho=np.load(f"{FEAT}/holdout_full.npy",mmap_mode='r')
yv=np.asarray(y[:n],dtype=np.float64); hm=np.asarray(ho[:n])
plon=np.asarray(X[:n,0],np.float64); plat=np.asarray(X[:n,1],np.float64); dlon=np.asarray(X[:n,2],np.float64); dlat=np.asarray(X[:n,3],np.float64)
def cell(lat,lon,dec):
    m=10**dec
    return (np.rint(lat*m).astype(np.int64))*(10**(dec+4))+(np.rint(lon*m).astype(np.int64))
def srl(lat,lon,dec):
    m=10**dec
    return (np.rint(lat*m).astype(np.int64))*10000+(np.rint(lon*m).astype(np.int64))
t0=time.time()
pu3=cell(plat,plon,3); do3=cell(dlat,dlon,3)
pu1=srl(plat,plon,1); do1=srl(dlat,dlon,1); rt1=pu1*np.int64(1000003)+do1
print("keys",time.time()-t0,flush=True)
folds=np.full(n,-1,dtype=np.int8); tr=np.where(hm==0)[0]
rng=np.random.default_rng(321); folds[tr]=rng.integers(0,5,size=len(tr)).astype(np.int8)
prior=float(yv[tr].mean()); ALPHA=20.0
out=np.empty((n,6),dtype=np.float32); UKS=[]; IST=R=[]
for j,(keys,name) in enumerate([(pu3,'pu3'),(do3,'do3'),(rt1,'rt1')]):
    uk,codes=np.unique(keys,return_inverse=True); K=len(uk); UKS.append(uk)
    istr=folds>=0; cs=codes[istr]; fs=folds[istr]; ys=yv[istr]
    sum_tot=np.bincount(cs,weights=ys,minlength=K); cnt_tot=np.bincount(cs,minlength=K)
    combo=cs.astype(np.int64)*5+fs.astype(np.int64)
    sum_f=np.bincount(combo,weights=ys,minlength=K*5).reshape(K,5); cnt_f=np.bincount(combo,minlength=K*5).reshape(K,5)
    s=sum_tot[codes].copy(); c=cnt_tot[codes].astype(np.float64); f=folds; m=f>=0
    s[m]-=sum_f[codes[m],f[m]]; c[m]-=cnt_f[codes[m],f[m]]
    out[:,2*j]=(s+prior*ALPHA)/(c+ALPHA); out[:,2*j+1]=np.log1p(cnt_tot[codes]).astype(np.float32)
    # save totals for test (store as npz later)
    UKS[j]=(uk,sum_tot,cnt_tot)
    print(name,"K",K,time.time()-t0,flush=True)
np.save(f"{FEAT}/TE_extra.npy",out)
Xt=np.load(f"{FEAT}/test_X.npy")
tplon=Xt[:,0].astype(np.float64); tplat=Xt[:,1].astype(np.float64); tdlon=Xt[:,2].astype(np.float64); tdlat=Xt[:,3].astype(np.float64)
tpu3=cell(tplat,tplon,3); tdo3=cell(tdlat,tdlon,3); tpu1=srl(tplat,tplon,1); tdo1=srl(tdlat,tdlon,1); trt1=tpu1*np.int64(1000003)+tdo1
teo=np.empty((len(Xt),6),dtype=np.float32)
for j,tk in enumerate([tpu3,tdo3,trt1]):
    uk,sum_tot,cnt_tot=UKS[j]; K=len(uk)
    pos=np.clip(np.searchsorted(uk,tk),0,K-1); found=uk[pos]==tk
    s=np.where(found,sum_tot[pos],0.0); c=np.where(found,cnt_tot[pos].astype(np.float64),0.0)
    teo[:,2*j]=(s+prior*ALPHA)/(c+ALPHA); teo[:,2*j+1]=np.log1p(c)
np.save(f"{FEAT}/test_TE_extra.npy",teo)
print("saved",out.shape,time.time()-t0,flush=True)
