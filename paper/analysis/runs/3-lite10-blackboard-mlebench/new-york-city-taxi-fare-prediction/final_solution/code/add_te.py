import numpy as np, pandas as pd, sys, time, zlib
sys.path.insert(0,'/download_job_code/payload/sandbox/workspaces/alkaid_phi')
import prep_data
OUT="/download_job_code/payload/sandbox/workspaces/alkaid_phi"
DATA="/download_job_code/payload/sandbox/data"
t0=time.time()
K=20.0

def derive(d):
    d=d.copy()
    d['is_jfk_pick']=(d.p_jfk<1.6).astype(np.int8); d['is_jfk_drop']=(d.d_jfk<1.6).astype(np.int8)
    d['is_lga_pick']=(d.p_lga<1.6).astype(np.int8); d['is_lga_drop']=(d.d_lga<1.6).astype(np.int8)
    d['is_ewr_pick']=(d.p_ewr<1.6).astype(np.int8); d['is_ewr_drop']=(d.d_ewr<1.6).astype(np.int8)
    d['is_airport']=((d.is_jfk_pick+d.is_jfk_drop+d.is_lga_pick+d.is_lga_drop+d.is_ewr_pick+d.is_ewr_drop)>0).astype(np.int8)
    d['log_hav']=np.log1p(d.haversine).astype(np.float32)
    d['is_night']=((d.hour>=20)|(d.hour<6)).astype(np.int8)
    d['is_peak']=((d.weekday<5)&(d.hour>=16)&(d.hour<20)).astype(np.int8)
    rad=np.radians
    for ang in (15,30,45):
        c=np.cos(rad(ang)); s=np.sin(rad(ang))
        d[f'pu_rot{ang}_x']=(d.pickup_longitude*c+d.pickup_latitude*s).astype(np.float32)
        d[f'pu_rot{ang}_y']=(d.pickup_latitude*c-d.pickup_longitude*s).astype(np.float32)
    return d

def cell(lat,lon,res):
    ci=np.floor((np.asarray(lat)+90)/res).astype(np.int64)
    cj=np.floor((np.asarray(lon)+180)/res).astype(np.int64)
    return ci*100000+cj

def route_code(pc,dc):
    return pc.astype(np.int64)*100000000+(dc%100000000)

def te_cols(idx,ncat,y,fold,tr_mask,prior):
    n=len(idx); out=np.full(n,prior,dtype=np.float32)
    for f in range(5):
        m_tr=tr_mask&(fold!=f); m_ap=tr_mask&(fold==f)
        s=np.bincount(idx[m_tr],weights=y[m_tr],minlength=ncat)
        c=np.bincount(idx[m_tr],minlength=ncat)
        mean=(s+prior*K)/(c+K)
        out[m_ap]=mean[idx[m_ap]]
    s=np.bincount(idx[tr_mask],weights=y[tr_mask],minlength=ncat)
    c=np.bincount(idx[tr_mask],minlength=ncat)
    mean=(s+prior*K)/(c+K)
    out[~tr_mask]=mean[idx[~tr_mask]]
    cnt=np.log1p(c)[idx].astype(np.float32)
    return out,cnt

def main():
    d=pd.read_parquet(f"{OUT}/train_feat.parquet")
    print("loaded",d.shape,time.time()-t0,flush=True)
    y=d.fare_amount.values.astype(np.float64)
    tr_mask=d.holdout.values==0
    prior=float(y[tr_mask].mean())
    fold=np.array([zlib.crc32(k.encode())%5 for k in d.key.values],dtype=np.int64)
    d=derive(d)
    specs={'p01':('pickup_latitude','pickup_longitude',0.01),'d01':('dropoff_latitude','dropoff_longitude',0.01),
           'p05':('pickup_latitude','pickup_longitude',0.05),'d05':('dropoff_latitude','dropoff_longitude',0.05)}
    tr_codes={}
    for name,(la,lo,res) in specs.items():
        tr_codes[name]=cell(d[la],d[lo],res)
    for name in ['p01','d01','p05','d05']:
        _,idx=np.unique(tr_codes[name],return_inverse=True); idx=idx.astype(np.int64)
        out,cnt=te_cols(idx,idx.max()+1,y,fold,tr_mask,prior)
        d[f'te_{name}']=out; d[f'cnt_{name}']=cnt
        print(name,'done',time.time()-t0,flush=True)
    tr_route=route_code(tr_codes['p01'],tr_codes['d01'])
    _,idx=np.unique(tr_route,return_inverse=True); idx=idx.astype(np.int64)
    out,cnt=te_cols(idx,idx.max()+1,y,fold,tr_mask,prior)
    d['te_route']=out; d['cnt_route']=cnt
    print('route done',time.time()-t0,flush=True)
    d.to_parquet(f"{OUT}/train_te.parquet",index=False)
    print("saved train_te",d.shape,time.time()-t0,flush=True)

    # ---- TEST ----
    trd=d[d.holdout==0]
    raw=pd.read_csv(f"{DATA}/test.csv")
    td=pd.DataFrame(prep_data.feat(raw)); td['key']=raw['key'].values
    flag=np.zeros(len(td),dtype=np.int8)
    for p in ['pickup','dropoff']:
        la=td[f'{p}_latitude'].values.copy(); lo=td[f'{p}_longitude'].values.copy()
        ok=(la>39)&(la<43)&(lo>-76)&(lo<-71)
        sw=(lo>39)&(lo<43)&(la>-76)&(la<-71)&(~ok)
        la2=np.where(sw,lo,la); lo2=np.where(sw,la,lo); flag=flag|sw.astype(np.int8)
        td[f'{p}_latitude']=la2.astype(np.float32); td[f'{p}_longitude']=lo2.astype(np.float32)
    td=derive(td); td['swap_flag']=flag
    la=td.pickup_latitude.values.astype(np.float64); lo=td.pickup_longitude.values.astype(np.float64)
    la2=td.dropoff_latitude.values.astype(np.float64); lo2=td.dropoff_longitude.values.astype(np.float64)
    td['haversine']=prep_data.hav(la,lo,la2,lo2).astype(np.float32)
    td['manhattan']=(prep_data.hav(la,lo,la,lo2)+prep_data.hav(la,lo2,la2,lo2)).astype(np.float32)
    td['log_hav']=np.log1p(td.haversine).astype(np.float32)
    td['pickup_latitude']=la.astype(np.float32); td['pickup_longitude']=lo.astype(np.float32)
    td['dropoff_latitude']=la2.astype(np.float32); td['dropoff_longitude']=lo2.astype(np.float32)
    ytr=trd.fare_amount.values.astype(np.float64)
    tr_codes_tr={name:cell(trd[la],trd[lo],res) for name,(la,lo,res) in specs.items()}
    tr_route_tr=route_code(tr_codes_tr['p01'],tr_codes_tr['d01'])
    def full_grouped(codes_tr,codes_te):
        g=pd.DataFrame({'c':codes_tr,'y':ytr}).groupby('c')['y'].agg(['sum','count'])
        mean=(g['sum']+prior*K)/(g['count']+K); cnt=np.log1p(g['count'])
        return codes_te.map(mean).fillna(prior).astype(np.float32).values, codes_te.map(cnt).fillna(0).astype(np.float32).values
    te_codes={}
    for name,(laN,loN,res) in specs.items():
        te_codes[name]=cell(td[laN],td[loN],res)
        a,b=full_grouped(tr_codes_tr[name],te_codes[name]); td[f'te_{name}']=a; td[f'cnt_{name}']=b
    te_route=route_code(te_codes['p01'],te_codes['d01'])
    a,b=full_grouped(tr_route_tr,te_route); td['te_route']=a; td['cnt_route']=b
    td.to_parquet(f"{OUT}/test_te.parquet",index=False)
    print("saved test_te",td.shape,"swapped",int(flag.sum()),time.time()-t0,flush=True)

if __name__=='__main__':
    main()
