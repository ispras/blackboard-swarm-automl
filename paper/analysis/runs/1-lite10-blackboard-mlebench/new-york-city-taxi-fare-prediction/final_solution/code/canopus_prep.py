import pandas as pd, numpy as np, os, sys, glob, time


DATA='/download_job_code/payload/sandbox/data'
OUT='/download_job_code/payload/sandbox/workspaces/canopus_pi/prep'
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT+'/parts', exist_ok=True)

R=6371.0088

def hav(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2=map(np.radians,[lon1,lat1,lon2,lat2])
    dlon=lon2-lon1; dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return (2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))).astype(np.float32)

def bearing(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2=map(np.radians,[lon1,lat1,lon2,lat2])
    dlon=lon2-lon1
    y=np.sin(dlon)*np.cos(lat2)
    x=np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return np.arctan2(y,x).astype(np.float32)

AIRPORTS={'jfk':(-73.7781,40.6413),'lga':(-73.8740,40.7769),'ewr':(-74.1745,40.6895)}
CTR=(-73.9857,40.7484)  # empire state-ish
THETA=np.radians(29.0)
CT,ST=np.cos(THETA),np.sin(THETA)

def feats(df):
    out=pd.DataFrame(index=df.index)
    plon=df['pickup_longitude'].to_numpy(np.float32); plat=df['pickup_latitude'].to_numpy(np.float32)
    dlon=df['dropoff_longitude'].to_numpy(np.float32); dlat=df['dropoff_latitude'].to_numpy(np.float32)
    out['plon']=plon; out['plat']=plat; out['dlon']=dlon; out['dlat']=dlat
    out['ddlon']=(dlon-plon).astype(np.float32); out['ddlat']=(dlat-plat).astype(np.float32)
    hd=hav(plon,plat,dlon,dlat)
    out['hav']=hd
    mlat=(plat+dlat)*0.5
    out['manh']=(hav(plon,plat,dlon,plat)+hav(plon,mlat,plon,dlat)).astype(np.float32)
    out['bear']=bearing(plon,plat,dlon,dlat)
    out['sin_bear']=np.sin(out['bear']).astype(np.float32); out['cos_bear']=np.cos(out['bear']).astype(np.float32)
    # rotated coords (manhattan grid)
    for name,lo,la in [('p',plon,plat),('d',dlon,dlat)]:
        out[name+'x']=(lo*CT+la*ST).astype(np.float32)
        out[name+'y']=(-lo*ST+la*CT).astype(np.float32)
    out['dx']=(out['dx']-out['px']).astype(np.float32); out['dy']=(out['dy']-out['py']).astype(np.float32)
    # airports
    for k,(alo,ala) in AIRPORTS.items():
        out['p_'+k]=hav(plon,plat,alo,ala)
        out['d_'+k]=hav(dlon,dlat,alo,ala)
    out['p_ctr']=hav(plon,plat,*CTR); out['d_ctr']=hav(dlon,dlat,*CTR)
    # temporal
    s=df['pickup_datetime'].astype(str).str.slice(0,19) if False else df['pickup_datetime']
    dt=pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S UTC', utc=True, errors='coerce')
    out['year']=dt.dt.year.astype(np.float32); out['month']=dt.dt.month.astype(np.float32)
    out['day']=dt.dt.day.astype(np.float32); out['weekday']=dt.dt.weekday.astype(np.float32)
    out['hour']=dt.dt.hour.astype(np.float32); out['minute']=dt.dt.minute.astype(np.float32)
    tod=out['hour'].to_numpy()+out['minute'].to_numpy()/60.0
    out['tod']=tod.astype(np.float32)
    out['tod_sin']=np.sin(2*np.pi*tod/24).astype(np.float32)
    out['tod_cos']=np.cos(2*np.pi*tod/24).astype(np.float32)
    out['is_weekend']=(out['weekday']>=5).astype(np.float32)
    out['passenger']=df['passenger_count'].to_numpy(np.float32)
    # grid cells for target encoding
    out['pcell']=(np.round(plat,2)*10000+np.round(plon,2)).astype(np.float32)
    out['dcell']=(np.round(dlat,2)*10000+np.round(dlon,2)).astype(np.float32)
    out['pcell3']=(np.round(plat,3)*100000+np.round(plon,3)).astype(np.float32)
    return out

def clean(df):
    fa=df['fare_amount']
    m=(fa>0)&(fa<=500)
    plon=df['pickup_longitude']; plat=df['pickup_latitude']
    dlon=df['dropoff_longitude']; dlat=df['dropoff_latitude']
    m&=plon.between(-75.5,-72.5)&dlon.between(-75.5,-72.5)
    m&=plat.between(40.0,42.0)&dlat.between(40.0,42.0)
    pc=df['passenger_count']; m&=(pc>=0)&(pc<=8)
    return df[m]

def process_test():
    t=pd.read_csv(DATA+'/test.csv')
    f=feats(t)
    f['key']=t['key'].values
    f.to_parquet(OUT+'/test.parquet', index=False)
    print('test', f.shape)

def process_holdout():
    reader=pd.read_csv(DATA+'/labels.csv', chunksize=3_000_000)
    offset=0; outs=[]
    for df in reader:
        df=df.copy(); df['orig_idx']=np.arange(offset,offset+len(df),dtype=np.int64); offset+=len(df)
        df=df[df['orig_idx']%100==0]
        df=clean(df)
        if len(df)==0: continue
        y=df['fare_amount'].to_numpy(np.float32); oi=df['orig_idx'].to_numpy(np.int64)
        f=feats(df); f['fare_amount']=y; f['orig_idx']=oi
        outs.append(f)
        print('holdout rows so far', sum(len(o) for o in outs), flush=True)
    h=pd.concat(outs, ignore_index=True)
    h.to_parquet(OUT+'/holdout.parquet', index=False)
    print('HOLDOUT', h.shape)

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'train'
    if mode=='holdout':
        process_holdout(); sys.exit()
    if mode=='test':
        process_test(); sys.exit()
    frac=float(sys.argv[2]) if len(sys.argv)>2 else 0.35
    seed=int(sys.argv[3]) if len(sys.argv)>3 else 42
    rng=np.random.default_rng(seed)
    reader=pd.read_csv(DATA+'/labels.csv', chunksize=3_000_000)
    nread=0; nsamp=0; part=0
    t0=time.time()
    offset=0
    for df in reader:
        nread+=len(df)
        df=df.copy(); df['orig_idx']=np.arange(offset, offset+len(df), dtype=np.int64)
        offset+=len(df)
        df=clean(df)
        keep=rng.random(len(df))<frac
        df=df[keep]
        if len(df)==0: continue
        y=df['fare_amount'].to_numpy(np.float32)
        oi=df['orig_idx'].to_numpy(np.int64)
        f=feats(df)
        f['fare_amount']=y; f['orig_idx']=oi
        f.to_parquet(f'{OUT}/parts/part_{part:04d}.parquet', index=False)
        part+=1; nsamp+=len(df)
        print(f'read={nread} kept={nsamp} parts={part} t={time.time()-t0:.0f}s', flush=True)
    print('DONE', nread, nsamp)
