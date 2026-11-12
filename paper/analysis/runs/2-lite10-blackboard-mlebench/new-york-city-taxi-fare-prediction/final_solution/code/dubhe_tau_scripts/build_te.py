import numpy as np, json, time
ROOT="/download_job_code/payload/sandbox"; FEAT=f"{ROOT}/workspaces/dubhe_tau/features"
meta=json.load(open(f"{FEAT}/meta.json")); n=meta['n_valid']
X=np.load(f"{FEAT}/X_full.npy",mmap_mode='r'); y=np.load(f"{FEAT}/y_full.npy",mmap_mode='r')
ho=np.load(f"{FEAT}/holdout_full.npy",mmap_mode='r')
yv=np.asarray(y[:n],dtype=np.float64); hm=np.asarray(ho[:n])
plon=np.asarray(X[:n,0],dtype=np.float64); plat=np.asarray(X[:n,1],dtype=np.float64)
dlon=np.asarray(X[:n,2],dtype=np.float64); dlat=np.asarray(X[:n,3],dtype=np.float64)
def cell(lat,lon,dec=2):
    m=10**dec
    return (np.rint(lat*m).astype(np.int64))*(10**(dec+4))+(np.rint(lon*m).astype(np.int64))
t0=time.time()
pu=cell(plat,plon); do=cell(dlat,dlon)
rt=pu*np.int64(1000003)+do
print("keys built",time.time()-t0,flush=True)
folds=np.full(n,-1,dtype=np.int8)
tr=np.where(hm==0)[0]
rng=np.random.default_rng(123); folds[tr]=rng.integers(0,5,size=len(tr)).astype(np.int8)
prior=float(yv[tr].mean()); ALPHA=20.0
out=np.empty((n,6),dtype=np.float32)
for j,(keys,name) in enumerate([(pu,'pu'),(do,'do'),(rt,'rt')]):
    uk,codes=np.unique(keys,return_inverse=True); K=len(uk)
    istr=folds>=0
    cs=codes[istr]; fs=folds[istr]; ys=yv[istr]
    sum_tot=np.bincount(codes[istr],weights=ys,minlength=K)
    cnt_tot=np.bincount(codes[istr],minlength=K)
    combo=cs.astype(np.int64)*5+fs.astype(np.int64)
    sum_f=np.bincount(combo,weights=ys,minlength=K*5).reshape(K,5)
    cnt_f=np.bincount(combo,minlength=K*5).reshape(K,5)
    s=sum_tot[codes].copy(); c=cnt_tot[codes].astype(np.float64)
    f=folds; m=f>=0
    s[m]-=sum_f[codes[m],f[m]]; c[m]-=cnt_f[codes[m],f[m]]
    out[:,2*j]=(s+prior*ALPHA)/(c+ALPHA)
    out[:,2*j+1]=np.log1p(cnt_tot[codes]).astype(np.float32)
    print(name,"K",K,"done",time.time()-t0,flush=True)
    if name=='pu': uk_pu=uk
    elif name=='do': uk_do=uk
    else: uk_rt=uk
np.save(f"{FEAT}/TE_dec2.npy",out)
# test
Xt=np.load(f"{FEAT}/test_X.npy")
def lookup(uk,keys):
    pos=np.searchsorted(uk,keys); pos=np.clip(pos,0,len(uk)-1)
    found=uk[pos]==keys
    return pos,found
tplon=Xt[:,0].astype(np.float64); tplat=Xt[:,1].astype(np.float64)
tdlon=Xt[:,2].astype(np.float64); tdlat=Xt[:,3].astype(np.float64)
tpu=cell(tplat,tplon); tdo=cell(tdlat,tdlon); trt=tpu*np.int64(1000003)+tdo
# recompute full-encoding tables (train only) for test: reuse bincounts by rebuilding quickly
# simpler: recompute using saved uk? we didn't save sums. Rebuild from keys via codes retrieval.
# Reload not possible; recompute per key type using uk + searchsorted into original keys is costly.
# Instead redo totals now:
def test_te(keys,uk):
    pos,found=lookup(uk,keys)
    return pos,found
# We need sum_tot/cnt_tot for test; recompute via bincount on codes of ALL rows mapped to uk
te_out=np.empty((len(Xt),6),dtype=np.float32)
for j,keys in enumerate([tpu,tdo,trt]):
    uk=uk_pu if j==0 else (uk_do if j==1 else uk_rt)
    # map test keys
    pos=np.searchsorted(uk,keys); pos=np.clip(pos,0,len(uk)-1); found=uk[pos]==keys
    # recompute totals mapping train keys to same uk
    if j==0: trkeys=pu
    elif j==1: trkeys=do
    else: trkeys=rt
    pos_tr=np.searchsorted(uk,trkeys)
    K=len(uk)
    sum_tot=np.bincount(pos_tr[istr],weights=yv[istr],minlength=K)
    cnt_tot=np.bincount(pos_tr[istr],minlength=K)
    s=sum_tot[pos]; c=cnt_tot[pos].astype(np.float64)
    s=np.where(found,s,0.0); c=np.where(found,c,0.0)
    te_out[:,2*j]=(s+prior*ALPHA)/(c+ALPHA)
    te_out[:,2*j+1]=np.log1p(c)
np.save(f"{FEAT}/test_TE_dec2.npy",te_out)
print("saved TE",out.shape,te_out.shape,time.time()-t0,flush=True)
