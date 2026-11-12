import pandas as pd, numpy as np, glob, os, gc, time, argparse
import lightgbm as lgb
W='/download_job_code/payload/sandbox/workspaces/canopus_pi'
PF=W+'/prep_full'
DATA='/download_job_code/payload/sandbox/data'
R=6371.0088
AIR={'jfk':(-73.7781,40.6413),'lga':(-73.8740,40.7769),'ewr':(-74.1745,40.6895)}
CTR=(-73.9857,40.7484); LOW=(-74.012,40.706)
def hav(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2=map(np.radians,[lon1,lat1,lon2,lat2])
    a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return (2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))).astype(np.float32)
def cell(la,lo):
    ilat=((la-40.0)*100).round(); ilon=((lo+74.0)*100).round()
    return (ilat.astype(np.int64)+1000)*100000+(ilon.astype(np.int64)+5000)

ap=argparse.ArgumentParser(); ap.add_argument('--build',action='store_true'); ap.add_argument('--train',action='store_true')
a=ap.parse_args()
t0=time.time()
if a.build:
    # TE maps from valid rows (prep_full)
    def build_map(col_lat,col_lon):
        S=None; C=None
        for f in sorted(glob.glob(PF+'/parts/part_*.parquet')):
            d=pd.read_parquet(f,columns=[col_lat,col_lon,'fare_amount'])
            c=pd.Series(cell(d[col_lat].to_numpy(),d[col_lon].to_numpy()))
            g=d['fare_amount'].groupby(c)
            s=g.sum(); n=g.count()
            S=s if S is None else S.add(s,fill_value=0); C=n if C is None else C.add(n,fill_value=0)
        return S,C
    Sp,Cp=build_map('plat','plon'); Sd,Cd=build_map('dlat','dlon')
    np.savez(W+'/corrupt_temaps.npz', p_sum=Sp.values,p_idx=Sp.index.values,p_cnt=Cp.values,
             d_sum=Sd.values,d_idx=Sd.index.values,d_cnt=Cd.values)
    print('te maps',len(Sp),len(Sd),f'{time.time()-t0:.0f}s',flush=True)
    def feats(df):
        o=pd.DataFrame(index=df.index)
        plon=df['pickup_longitude'].to_numpy(np.float64); plat=df['pickup_latitude'].to_numpy(np.float64)
        dlon=df['dropoff_longitude'].to_numpy(np.float64); dlat=df['dropoff_latitude'].to_numpy(np.float64)
        pok_l=(plon>=-75.5)&(plon<=-72.5); pok_a=(plat>=40)&(plat<=42)
        dok_l=(dlon>=-75.5)&(dlon<=-72.5); dok_a=(dlat>=40)&(dlat<=42)
        # transposition recovery
        psw=(~pok_l)&(~pok_a)&(plat>=-75.5)&(plat<=-72.5)&(plon>=40)&(plon<=42)
        dsw=(~dok_l)&(~dok_a)&(dlat>=-75.5)&(dlat<=-72.5)&(dlon>=40)&(dlon<=42)
        plon2=np.where(psw,plat,plon); plat2=np.where(psw,plon,plat)
        dlon2=np.where(dsw,dlat,dlon); dlat2=np.where(dsw,dlon,dlat)
        pok_l2=(plon2>=-75.5)&(plon2<=-72.5); pok_a2=(plat2>=40)&(plat2<=42)
        dok_l2=(dlon2>=-75.5)&(dlon2<=-72.5); dok_a2=(dlat2>=40)&(dlat2<=42)
        pv=pok_l2&pok_a2; dv=dok_l2&dok_a2
        o['p_lat_ok']=pok_a2.astype(np.float32); o['p_lon_ok']=pok_l2.astype(np.float32)
        o['d_lat_ok']=dok_a2.astype(np.float32); o['d_lon_ok']=dok_l2.astype(np.float32)
        o['p_swap']=psw.astype(np.float32); o['d_swap']=dsw.astype(np.float32)
        o['p_valid']=pv.astype(np.float32); o['d_valid']=dv.astype(np.float32)
        o['both_zero']=((plon==0)&(plat==0)&(dlon==0)&(dlat==0)).astype(np.float32)
        o['plat']=np.where(pok_a2,plat2,np.nan).astype(np.float32)
        o['plon']=np.where(pok_l2,plon2,np.nan).astype(np.float32)
        o['dlat']=np.where(dok_a2,dlat2,np.nan).astype(np.float32)
        o['dlon']=np.where(dok_l2,dlon2,np.nan).astype(np.float32)
        for nm,(lo,la) in AIR.items():
            o['p_'+nm]=np.where(pv,hav(plon2,plat2,lo,la),np.nan).astype(np.float32)
            o['d_'+nm]=np.where(dv,hav(dlon2,dlat2,lo,la),np.nan).astype(np.float32)
        o['p_ctr']=np.where(pv,hav(plon2,plat2,*CTR),np.nan).astype(np.float32)
        o['d_ctr']=np.where(dv,hav(dlon2,dlat2,*CTR),np.nan).astype(np.float32)
        o['p_low']=np.where(pv,hav(plon2,plat2,*LOW),np.nan).astype(np.float32)
        o['d_low']=np.where(dv,hav(dlon2,dlat2,*LOW),np.nan).astype(np.float32)
        return o,pv,dv,plon2,plat2,dlon2,dlat2
    out=[]
    reader=pd.read_csv(DATA+'/labels.csv',chunksize=3_000_000)
    off=0
    for df in reader:
        df=df.copy(); df['orig_idx']=np.arange(off,off+len(df),dtype=np.int64); off+=len(df)
        m=~((df.pickup_longitude.between(-75.5,-72.5))&(df.pickup_latitude.between(40,42))&
            (df.dropoff_longitude.between(-75.5,-72.5))&(df.dropoff_latitude.between(40,42)))
        m&=df.fare_amount.between(2.5,500)&df.passenger_count.between(0,8)
        df=df[m]
        if len(df)==0: continue
        f,pv,dv,pl2,pa2,dl2,da2=feats(df)
        f['year']=df.pickup_datetime.str.slice(0,4).astype(np.float32).values
        f['month']=df.pickup_datetime.str.slice(5,7).astype(np.float32).values
        f['day']=df.pickup_datetime.str.slice(8,10).astype(np.float32).values
        hh=df.pickup_datetime.str.slice(11,13).astype(np.float32).values
        mm=df.pickup_datetime.str.slice(14,16).astype(np.float32).values
        f['hour']=hh; f['minute']=mm
        tod=hh+mm/60.0
        f['tod_sin']=np.sin(2*np.pi*tod/24).astype(np.float32); f['tod_cos']=np.cos(2*np.pi*tod/24).astype(np.float32)
        dt=pd.to_datetime(df.pickup_datetime.str.slice(0,19),format='%Y-%m-%d %H:%M:%S',errors='coerce')
        f['weekday']=dt.dt.weekday.values.astype(np.float32)
        f['passenger']=df.passenger_count.values.astype(np.float32)
        f['fare_amount']=df.fare_amount.values.astype(np.float32)
        f['orig_idx']=df.orig_idx.values
        out.append(f)
    D=pd.concat(out,ignore_index=True)
    # TE for valid endpoints
    tm=np.load(W+'/corrupt_temaps.npz')
    prior=float(tm['p_sum'].sum()/tm['p_cnt'].sum())
    for pre,latc,lonc in [('p','plat','plon'),('d','dlat','dlon')]:
        s=np.full(len(D),np.nan,dtype=np.float32); c=np.zeros(len(D),dtype=np.float32)
        for sub,idx,sarr,carr in [(pre+'idx',None,None,None)]:
            pass
        # map only where both coords valid
        ok=D[latc].notna().values&D[lonc].notna().values
        if ok.any():
            cc=cell(D[latc].values[ok],D[lonc].values[ok])
            S=pd.Series(tm[pre+'_sum'],index=tm[pre+'_idx']); C=pd.Series(tm[pre+'_cnt'],index=tm[pre+'_idx'])
            ssum=S.reindex(cc).values; ccount=C.reindex(cc).values
            s[ok]=((ssum+prior*20)/(ccount+20)).astype(np.float32)
            c[ok]=np.log1p(np.nan_to_num(ccount)).astype(np.float32)
        D['te_'+pre]=s; D['cnt_'+pre]=c
    D.to_parquet(W+'/corrupt_train2.parquet',index=False)
    print('corrupt train',D.shape,f'{time.time()-t0:.0f}s',flush=True)
    # test corrupt features
    tp=pd.read_parquet(W+'/prep/test.parquet')
    bad=~((tp.plon.between(-75.5,-72.5))&(tp.plat.between(40,42))&(tp.dlon.between(-75.5,-72.5))&(tp.dlat.between(40,42)))
    tt=tp[bad].copy()
    # reuse test.parquet raw coords with same feature builder via temp df
    tmp=pd.DataFrame({'pickup_longitude':tt.plon.values,'pickup_latitude':tt.plat.values,
                      'dropoff_longitude':tt.dlon.values,'dropoff_latitude':tt.dlat.values})
    f,pv,dv,pl2,pa2,dl2,da2=feats(tmp)
    f['year']=tt.year.values; f['month']=tt.month.values; f['day']=tt.day.values
    f['hour']=tt.hour.values; f['minute']=tt.minute.values
    tod=tt.tod.values; f['tod_sin']=tt.tod_sin.values; f['tod_cos']=tt.tod_cos.values
    f['weekday']=tt.weekday.values; f['passenger']=tt.passenger.values
    for pre,latc,lonc in [('p','plat','plon'),('d','dlat','dlon')]:
        s=np.full(len(f),np.nan,dtype=np.float32); c=np.zeros(len(f),dtype=np.float32)
        ok=f[latc].notna().values&f[lonc].notna().values
        if ok.any():
            cc=cell(f[latc].values[ok],f[lonc].values[ok])
            S=pd.Series(tm[pre+'_sum'],index=tm[pre+'_idx']); C=pd.Series(tm[pre+'_cnt'],index=tm[pre+'_idx'])
            ssum=S.reindex(cc).values; ccount=C.reindex(cc).values
            s[ok]=((ssum+prior*20)/(ccount+20)).astype(np.float32)
            c[ok]=np.log1p(np.nan_to_num(ccount)).astype(np.float32)
        f['te_'+pre]=s; f['cnt_'+pre]=c
    f['key']=tt.key.values
    f.to_parquet(W+'/corrupt_test2.parquet',index=False)
    print('corrupt test',f.shape,flush=True)
