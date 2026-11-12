import numpy as np, pyarrow as pa, pyarrow.parquet as pq, time
ROOT='/download_job_code/payload/sandbox'; W=f'{ROOT}/workspaces/antares_omicron'
t0=time.time()
LON_MIN,LAT_MIN=-75.6,39.4
PRIOR=11.3; K=20.0
GRIDS=[('01',0.01,400,400),('05',0.05,80,80)]

def cells(lon,lat,g,nlon,nlat):
    ok=np.isfinite(lon)&np.isfinite(lat)
    il=np.clip(((np.nan_to_num(lon,nan=LON_MIN)-LON_MIN)/g).astype(np.int64),0,nlon-1)
    ia=np.clip(((np.nan_to_num(lat,nan=LAT_MIN)-LAT_MIN)/g).astype(np.int64),0,nlat-1)
    return (il*nlat+ia).astype(np.int64),ok

def sums(code,ok,fare,ncell):
    c=code.copy(); c[~ok]=0
    fv=np.where(ok,fare,0.0)
    s=np.bincount(c,weights=fv,minlength=ncell)
    n=np.bincount(c,minlength=ncell)
    return s,n

def main():
    tr=pq.read_table(f'{W}/feat/train_feats.parquet',columns=['fare','lon1','lat1','lon2','lat2'])
    te=pq.read_table(f'{W}/feat/test_feats.parquet',columns=['lon1','lat1','lon2','lat2'])
    fare=tr['fare'].to_numpy().astype(np.float64); n=len(fare)
    tlon1=tr['lon1'].to_numpy(); tlat1=tr['lat1'].to_numpy(); tlon2=tr['lon2'].to_numpy(); tlat2=tr['lat2'].to_numpy()
    vlon1=te['lon1'].to_numpy(); vlat1=te['lat1'].to_numpy(); vlon2=te['lon2'].to_numpy(); vlat2=te['lat2'].to_numpy()
    fold=(np.arange(n)%5).astype(np.int8)
    outtr={}; outte={}
    def add(name,code,ok,ncell,tr_code,tr_ok):
        s,nn=sums(code,ok,fare,ncell)
        tev=((s+PRIOR*K)/(nn+K)).astype(np.float32)
        cnt=nn.astype(np.float32)
        # OOF te for train
        oof=np.full(n,np.nan,dtype=np.float32)
        for f in range(5):
            m=(fold==f)&ok
            if not m.any(): continue
            s_f,n_f=sums(code[m],ok[m],fare[m],ncell)
            o=(s-s_f); c=(nn-n_f)
            tv=((o+PRIOR*K)/(c+K)).astype(np.float32)
            oof[m]=tv[code[m]]
        # test map
        tv_te=np.full(len(tr_code),np.nan,dtype=np.float32); tv_te[tr_ok]=tev[tr_code[tr_ok]]
        tv_cnt=np.full(len(tr_code),np.nan,dtype=np.float32); tv_cnt[tr_ok]=cnt[tr_code[tr_ok]]
        outtr[f'te_{name}']=oof; outtr[f'cnt_{name}']=np.where(ok,cnt[code],np.nan).astype(np.float32)
        outte[f'te_{name}']=tv_te; outte[f'cnt_{name}']=tv_cnt
        return tev,cnt
    for name,g,nlon,nlat in GRIDS:
        nc=nlon*nlat
        pc,pok=cells(tlon1,tlat1,g,nlon,nlat); dc,dok=cells(tlon2,tlat2,g,nlon,nlat)
        vpc,vpok=cells(vlon1,vlat1,g,nlon,nlat); vdc,vdok=cells(vlon2,vlat2,g,nlon,nlat)
        add(f'p{name}',pc,pok,nc,vpc,vpok)
        add(f'd{name}',dc,dok,nc,vdc,vdok)
        print('grid',name,time.time()-t0,flush=True)
        if name=='05':
            rc=pc*nc+dc; rok=pok&dok; vrc=vpc*nc+vdc; vrok=vpok&vdok
            add('route05',rc,rok,nc*nc,vrc,vrok)
            outtr['same_cell05']=((pc==dc)&rok).astype(np.float32)
            outte['same_cell05']=((vpc==vdc)&vrok).astype(np.float32)
            print('route',time.time()-t0,flush=True)
    pq.write_table(pa.table({k:pa.array(v) for k,v in outtr.items()}),f'{W}/feat/te_feats.parquet',compression='zstd')
    pq.write_table(pa.table({k:pa.array(v) for k,v in outte.items()}),f'{W}/feat/te_test.parquet',compression='zstd')
    print('saved',time.time()-t0,list(outtr.keys()),flush=True)

if __name__=='__main__':
    main()
