import numpy as np, json, time
ROOT="/download_job_code/payload/sandbox"; FEAT=f"{ROOT}/workspaces/dubhe_tau/features"
meta=json.load(open(f"{FEAT}/meta.json")); n=meta['n_valid']
X=np.load(f"{FEAT}/X_full.npy",mmap_mode='r'); y=np.load(f"{FEAT}/y_full.npy",mmap_mode='r'); ho=np.load(f"{FEAT}/holdout_full.npy",mmap_mode='r')
yv=np.asarray(y[:n],np.float64); hm=np.asarray(ho[:n])
plon=np.asarray(X[:n,0],np.float64); plat=np.asarray(X[:n,1],np.float64); dlon=np.asarray(X[:n,2],np.float64); dlat=np.asarray(X[:n,3],np.float64)
hav=np.asarray(X[:n,12],np.float64)
def cell(lat,lon,dec):
    m=10**dec; return (np.rint(lat*m).astype(np.int64))*(10**(dec+4))+(np.rint(lon*m).astype(np.int64))
t0=time.time(); tr=np.where(hm==0)[0]
# distance baseline by 0.2 km bins
BINW=0.2; NB=600
bins=np.clip((hav/BINW).astype(np.int64),0,NB-1)
glob=float(yv[tr].mean()); K=200.0
sum_b=np.bincount(bins[tr],weights=yv[tr],minlength=NB); cnt_b=np.bincount(bins[tr],minlength=NB)
base_b=(sum_b+glob*K)/(cnt_b+K)
base_row=base_b[bins]
resid=yv-base_row
print("baseline built",time.time()-t0,flush=True)
folds=np.full(n,-1,np.int8); rng=np.random.default_rng(777); folds[tr]=rng.integers(0,5,size=len(tr)).astype(np.int8)
pu=cell(plat,plon,2); do=cell(dlat,dlon,2); rt=pu*np.int64(1000003)+do
out=np.empty((n,4),dtype=np.float32); out[:,3]=base_row.astype(np.float32)
ALPHA=20.0
for j,(keys,name) in enumerate([(pu,'pu'),(do,'do'),(rt,'rt')]):
    uk,codes=np.unique(keys,return_inverse=True); Kk=len(uk)
    istr=folds>=0; cs=codes[istr]; fs=folds[istr]; rs=resid[istr]
    sum_tot=np.bincount(cs,weights=rs,minlength=Kk); cnt_tot=np.bincount(cs,minlength=Kk)
    combo=cs.astype(np.int64)*5+fs.astype(np.int64)
    sum_f=np.bincount(combo,weights=rs,minlength=Kk*5).reshape(Kk,5); cnt_f=np.bincount(combo,minlength=Kk*5).reshape(Kk,5)
    s=sum_tot[codes].copy(); c=cnt_tot[codes].astype(np.float64); f=folds; m=f>=0
    s[m]-=sum_f[codes[m],f[m]]; c[m]-=cnt_f[codes[m],f[m]]
    out[:,j]=(s/(c+ALPHA)).astype(np.float32)
    print(name,"K",Kk,time.time()-t0,flush=True)
    if name=='pu': uk_pu=uk; st_pu=sum_tot; ct_pu=cnt_tot
    elif name=='do': uk_do=uk; st_do=sum_tot; ct_do=cnt_tot
    else: uk_rt=uk; st_rt=sum_tot; ct_rt=cnt_tot
np.save(f"{FEAT}/TE_res.npy",out)
Xt=np.load(f"{FEAT}/test_X.npy")
thav=np.asarray(Xt[:,12],np.float64); tb=np.clip((thav/BINW).astype(np.int64),0,NB-1)
tbase=base_b[tb]
def tcell(lat,lon,dec):
    m=10**dec; return (np.rint(lat*m).astype(np.int64))*(10**(dec+4))+(np.rint(lon*m).astype(np.int64))
tpu=tcell(Xt[:,1].astype(np.float64),Xt[:,0].astype(np.float64),2); tdo=tcell(Xt[:,3].astype(np.float64),Xt[:,2].astype(np.float64),2); trt=tpu*np.int64(1000003)+tdo
to=np.empty((len(Xt),4),dtype=np.float32); to[:,3]=tbase.astype(np.float32)
for j,(uk,st,ct,tk) in enumerate([(uk_pu,st_pu,ct_pu,tpu),(uk_do,st_do,ct_do,tdo),(uk_rt,st_rt,ct_rt,trt)]):
    pos=np.clip(np.searchsorted(uk,tk),0,len(uk)-1); found=uk[pos]==tk
    s=np.where(found,st[pos],0.0); c=np.where(found,ct[pos].astype(np.float64),0.0)
    to[:,j]=(s/(c+ALPHA)).astype(np.float32)
np.save(f"{FEAT}/test_TE_res.npy",to)
print("saved",out.shape,to.shape,time.time()-t0,flush=True)
