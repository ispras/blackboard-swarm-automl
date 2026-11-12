import numpy as np, pandas as pd, pyarrow as pa, pyarrow.csv as pacsv, pyarrow.compute as pc, pyarrow.parquet as pq, time, os, re

ROOT='/download_job_code/payload/sandbox'
OUT=f'{ROOT}/workspaces/antares_omicron/feat'
os.makedirs(OUT, exist_ok=True)
HOLD=277

def hav(lon1,lat1,lon2,lat2):
    R=6371.0
    p1=np.radians(lat1); p2=np.radians(lat2)
    dphi=p2-p1; dl=np.radians(lon2-lon1)
    a=np.sin(dphi/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return (2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))).astype(np.float32)


LON_B=(-75.5,-72.5); LAT_B=(39.5,42.0)
def plausible(lon,lat):
    return (lon>=LON_B[0])&(lon<=LON_B[1])&(lat>=LAT_B[0])&(lat<=LAT_B[1])
def repair(lon,lat):
    lon=lon.astype(np.float32); lat=lat.astype(np.float32)
    missing=(lon==0)&(lat==0)
    lon=np.where(missing,np.float32(np.nan),lon)
    lat=np.where(missing,np.float32(np.nan),lat)
    ok=plausible(lon,lat)
    sw=plausible(lat,lon)
    swap=(~ok)&sw&(~missing)
    lon2=np.where(swap,lat,lon).astype(np.float32)
    lat2=np.where(swap,lon,lat).astype(np.float32)
    has=(~missing).astype(np.int8)
    return lon2,lat2,swap.astype(np.int8),has

POI={'jfk':(-73.7781,40.6413),'lga':(-73.8740,40.7769),'ewr':(-74.1745,40.6895),
     'midtown':(-73.9855,40.7580),'dt':(-74.0113,40.7069)}

def compute(lon1,lat1,lon2,lat2,pcnt,y,mo,d,h,mi,sec,ns):
    lon1=lon1.astype(np.float32); lat1=lat1.astype(np.float32)
    lon2=lon2.astype(np.float32); lat2=lat2.astype(np.float32)
    feats={}
    feats['lon1']=lon1; feats['lat1']=lat1; feats['lon2']=lon2; feats['lat2']=lat2
    feats['passenger_count']=pcnt.astype(np.int16)
    y=y.astype(np.int64); mo=mo.astype(np.int64); d=d.astype(np.int64)
    h=h.astype(np.int16); mi=mi.astype(np.int16)
    feats['year']=y.astype(np.int16); feats['month']=mo.astype(np.int8); feats['day']=d.astype(np.int8)
    feats['hour']=h; feats['minute']=mi
    sec=sec.astype(np.int64); ns=ns.astype(np.int64)
    feats['sec']=sec.astype(np.int16)
    yy2=y-(mo<=2)
    era=np.where(yy2>=0, yy2//400, (yy2-399)//400)
    yoe=yy2-era*400
    mp=np.where(mo>2, mo-3, mo+9)
    doy0=(153*mp+2)//5+d-1
    doe=yoe*365+yoe//4-yoe//100+doy0
    days=era*146097+doe-719468
    feats['dow']=((days+3)%7).astype(np.int8)
    cum=np.array([0,31,59,90,120,151,181,212,243,273,304,334],dtype=np.int64)
    leap=((y%4==0)&((y%100!=0)|(y%400==0))).astype(np.int64)
    feats['doy']=(cum[mo-1]+d+np.where(mo>2,leap,0)).astype(np.int16)
    feats['is_weekend']=(feats['dow']>=5).astype(np.int8)
    feats['dist_km']=hav(lon1,lat1,lon2,lat2)
    feats['dlat']=(lat2-lat1).astype(np.float32); feats['dlon']=(lon2-lon1).astype(np.float32)
    feats['abs_dlat']=np.abs(feats['dlat']); feats['abs_dlon']=np.abs(feats['dlon'])
    mlat=feats['abs_dlat']*111.0
    mlon=feats['abs_dlon']*111.0*np.cos(np.radians((lat1+lat2)/2))
    feats['manhattan_km']=(mlat+mlon).astype(np.float32)
    yy=np.sin(np.radians(lon2-lon1))*np.cos(np.radians(lat2))
    xx=np.cos(np.radians(lat1))*np.sin(np.radians(lat2))-np.sin(np.radians(lat1))*np.cos(np.radians(lat2))*np.cos(np.radians(lon2-lon1))
    feats['bearing']=np.degrees(np.arctan2(yy,xx)).astype(np.float32)
    for name,(lo,la) in POI.items():
        d1=hav(lon1,lat1,lo,la); d2=hav(lon2,lat2,lo,la)
        feats[f'p_{name}']=d1; feats[f'd_{name}']=d2
        feats[f'min_{name}']=np.minimum(d1,d2)
    feats['ts_days']=(days.astype(np.float64)+(h*3600.0+mi*60.0+sec)/86400.0).astype(np.float32)
    hf=(h+mi/60.0).astype(np.float32)
    feats['hour_sin']=np.sin(2*np.pi*hf/24).astype(np.float32)
    feats['hour_cos']=np.cos(2*np.pi*hf/24).astype(np.float32)
    feats['dow_sin']=np.sin(2*np.pi*feats['dow']/7).astype(np.float32)
    feats['dow_cos']=np.cos(2*np.pi*feats['dow']/7).astype(np.float32)
    dk=feats['dist_km']; mk=feats['manhattan_km']
    feats['log_dist']=np.log1p(np.abs(dk)).astype(np.float32)
    feats['euclid_deg']=np.sqrt(feats['dlat']**2+feats['dlon']**2).astype(np.float32)
    feats['manh_ratio']=(mk/(dk+1e-3)).astype(np.float32)
    feats['dist_ratio']=(dk/(mk+1e-3)).astype(np.float32)
    br=feats['bearing']
    feats['bearing_sin']=np.sin(np.radians(br)).astype(np.float32)
    feats['bearing_cos']=np.cos(np.radians(br)).astype(np.float32)
    pairs=[('jfk','lga'),('jfk','ewr'),('lga','ewr')]
    pmin=np.minimum(np.minimum(feats['p_jfk'],feats['p_lga']),feats['p_ewr'])
    dmin=np.minimum(np.minimum(feats['d_jfk'],feats['d_lga']),feats['d_ewr'])
    feats['p_air_min']=pmin; feats['d_air_min']=dmin
    feats['is_airport_trip']=(np.minimum(pmin,dmin)<2.0).astype(np.int8)
    seed=(((((y*13+mo)*32+d)*24+h)*60+mi)*60+sec)*1024+ns
    x=seed.astype(np.uint64)
    x=(x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x=(x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x=x ^ (x >> np.uint64(31))
    feats['holdout']=((x % np.uint64(200))==np.uint64(0))
    return feats

def parse_time(s):
    dt=pc.cast(s, pa.string())
    def sub(off,ln): return pc.cast(pc.utf8_slice_codeunits(dt, off, off+ln), pa.int64()).to_numpy(zero_copy_only=False)
    return sub(0,4),sub(5,2),sub(8,2),sub(11,2),sub(14,2),sub(17,2)

def parse_key_suffix(keys):
    k=pc.cast(keys, pa.string())
    ex=pc.extract_regex(k, pattern=r'\.(?P<x>\d+)$')
    return pc.cast(pc.struct_field(ex,'x'), pa.int64()).to_numpy(zero_copy_only=False)

def build_train():
    t0=time.time()
    cols=['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
    tbl=pacsv.read_csv(f'{ROOT}/data/labels.csv', convert_options=pacsv.ConvertOptions(include_columns=cols))
    print('read',tbl.num_rows,time.time()-t0,flush=True)
    y,mo,d,h,mi,sec=parse_time(tbl['pickup_datetime'])
    ks=parse_key_suffix(tbl['key'])
    f=lambda c,t: pc.cast(tbl[c],t).to_numpy(zero_copy_only=False)
    plon,plat,psw,phsv=repair(f('pickup_longitude',pa.float32()),f('pickup_latitude',pa.float32()))
    dlon,dlat_,dsw,dhsv=repair(f('dropoff_longitude',pa.float32()),f('dropoff_latitude',pa.float32()))
    feats=compute(plon,plat,dlon,dlat_,f('passenger_count',pa.int16()),y,mo,d,h,mi,sec,ks)
    feats['swap_flag']=(psw|dsw).astype(np.int8)
    feats['has_pickup']=phsv; feats['has_dropoff']=dhsv
    feats['fare']=f('fare_amount',pa.float32())
    feats['key_suffix']=ks
    del tbl
    print('feats',time.time()-t0,flush=True)
    tab=pa.table({k:pa.array(v) for k,v in feats.items()})
    pq.write_table(tab,f'{OUT}/train_feats.parquet',compression='zstd')
    print('saved train',tab.num_rows,'holdout',int(feats['holdout'].sum()),time.time()-t0,flush=True)

def build_test():
    t0=time.time()
    tbl=pacsv.read_csv(f'{ROOT}/data/test.csv')
    y,mo,d,h,mi,sec=parse_time(tbl['pickup_datetime'])
    ks=parse_key_suffix(tbl['key'])
    f=lambda c,t: pc.cast(tbl[c],t).to_numpy(zero_copy_only=False)
    plon,plat,psw,phsv=repair(f('pickup_longitude',pa.float32()),f('pickup_latitude',pa.float32()))
    dlon,dlat_,dsw,dhsv=repair(f('dropoff_longitude',pa.float32()),f('dropoff_latitude',pa.float32()))
    feats=compute(plon,plat,dlon,dlat_,f('passenger_count',pa.int16()),y,mo,d,h,mi,sec,ks)
    feats['swap_flag']=(psw|dsw).astype(np.int8)
    feats['has_pickup']=phsv; feats['has_dropoff']=dhsv
    feats['key_suffix']=ks
    feats['key']=pc.cast(tbl['key'],pa.string()).to_numpy(zero_copy_only=False)
    tab=pa.table({k:pa.array(v) for k,v in feats.items()})
    pq.write_table(tab,f'{OUT}/test_feats.parquet',compression='zstd')
    print('saved test',tab.num_rows,time.time()-t0,flush=True)

if __name__=='__main__':
    build_train(); build_test()
