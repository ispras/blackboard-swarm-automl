import pandas as pd, numpy as np, glob, os, argparse, gc, time
import lightgbm as lgb
P='/download_job_code/payload/sandbox/workspaces/canopus_pi/prep'
W='/download_job_code/payload/sandbox/workspaces/canopus_pi'
ap=argparse.ArgumentParser()
ap.add_argument('--tag', type=str, default='d19m')
ap.add_argument('--prepdir', type=str, default=P)
ap.add_argument('--maxrows', type=int, default=19_000_000)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--threads', type=int, default=8)
args=ap.parse_args()
t0=time.time()
model=lgb.Booster(model_file=f'{W}/models/lgb_{args.tag}.txt')
feats=list(np.load(f'{W}/models/lgb_{args.tag}_feats.npy'))

def add_extra(d):
    d['p_air_min']=d[['p_jfk','p_lga','p_ewr']].min(axis=1).astype(np.float32)
    d['d_air_min']=d[['d_jfk','d_lga','d_ewr']].min(axis=1).astype(np.float32)
    d['air_trip']=((d['p_air_min']<2)|(d['d_air_min']<2)).astype(np.float32)
    d['jfk_manh']=(((d['p_jfk']<2)&(d['d_ctr']<8))|((d['d_jfk']<2)&(d['p_ctr']<8))).astype(np.float32)
    d['hav_per_dlon']=(d['hav']/(np.abs(d['ddlon'])+1e-3)).astype(np.float32)
    d['manh_minus_hav']=(d['manh']-d['hav']).astype(np.float32)
    return d
def codes(d):
    ilat=((d['plat'].to_numpy().astype(np.float64)-40.0)*100.0).round().astype(np.int64)
    ilon=((d['plon'].to_numpy().astype(np.float64)+74.0)*100.0).round().astype(np.int64)
    jlat=((d['dlat'].to_numpy().astype(np.float64)-40.0)*100.0).round().astype(np.int64)
    jlon=((d['dlon'].to_numpy().astype(np.float64)+74.0)*100.0).round().astype(np.int64)
    pc=(ilat+1000)*100000+(ilon+5000)
    dc=(jlat+1000)*100000+(jlon+5000)
    klat=((d['plat'].to_numpy().astype(np.float64)-40.0)*1000.0).round().astype(np.int64)
    klon=((d['plon'].to_numpy().astype(np.float64)+74.0)*1000.0).round().astype(np.int64)
    pc3=(klat+10000)*100000+(klon+50000)
    return pc,dc,pc3,pc*1_000_000_000+dc

# rebuild exact training sample for TE maps
files=sorted(glob.glob(args.prepdir+'/parts/part_*.parquet'))
df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df=df[(df['fare_amount']>=2.5)&(df['orig_idx'].to_numpy()%100!=0)].reset_index(drop=True); gc.collect()
if len(df)>args.maxrows: df=df.sample(n=args.maxrows, random_state=args.seed).reset_index(drop=True)
df=add_extra(df)
pc,dc,pc3,od=codes(df); df['_pc']=pc; df['_dc']=dc; df['_pc3']=pc3; df['_od']=od
del pc,dc,pc3,od; gc.collect()
y=df['fare_amount'].to_numpy(np.float32); prior=float(y.mean())
te_maps={}
for key,m in [('_pc',20.0),('_dc',20.0),('_od',10.0),('_pc3',10.0)]:
    g=df.groupby(key)['fare_amount']; s=g.sum(); c=g.count()
    te_maps[key]=( ((s+prior*m)/(c+m)).to_dict(), c.to_dict() )
print('TE maps built',f'{time.time()-t0:.0f}s',flush=True)
del df; gc.collect()

t=pd.read_parquet(P+'/test.parquet')
t=add_extra(t); pc,dc,pc3,od=codes(t); t['_pc']=pc; t['_dc']=dc; t['_pc3']=pc3; t['_od']=od
for key,m in [('_pc',20.0),('_dc',20.0),('_od',10.0),('_pc3',10.0)]:
    te,cp=te_maps[key]
    t[key+'_te']=t[key].map(te).fillna(prior).to_numpy(np.float32)
    t[key+'_cnt']=np.log1p(t[key].map(cp).fillna(0).to_numpy()).astype(np.float32)
X=t[feats].to_numpy(np.float32)
pred=model.predict(X, num_iteration=model.best_iteration)
pred=np.clip(pred,2.5,500.0)
valid=((t.plon.between(-75.5,-72.5))&(t.plat.between(40,42))&
       (t.dlon.between(-75.5,-72.5))&(t.dlat.between(40,42))).to_numpy()
os.makedirs(f'{W}/preds',exist_ok=True)
np.save(f'{W}/preds/test_pred_{args.tag}.npy', pred)
np.save(f'{W}/preds/test_validmask_{args.tag}.npy', valid)
sub=pd.DataFrame({'key':t['key'].values,'fare_amount':pred})
sub.to_csv(f'{W}/preds/submission_{args.tag}.csv', index=False)
print('saved test preds', len(pred), 'valid', valid.sum(), 'corrupt', (~valid).sum(), f'{time.time()-t0:.0f}s')
