import pandas as pd, numpy as np, glob, os, argparse, gc, time
import lightgbm as lgb
P='/download_job_code/payload/sandbox/workspaces/canopus_pi/prep'
ap=argparse.ArgumentParser()
ap.add_argument('--maxrows', type=int, default=19_000_000)
ap.add_argument('--rounds', type=int, default=4000)
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--leaves', type=int, default=255)
ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--tag', type=str, default='v3')
ap.add_argument('--prepdir', type=str, default=P)
ap.add_argument('--no_te', action='store_true')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--min_data', type=int, default=40)
ap.add_argument('--patience', type=int, default=200)
ap.add_argument('--ff', type=float, default=0.85)
ap.add_argument('--bf', type=float, default=0.8)
ap.add_argument('--l2', type=float, default=1.0)
ap.add_argument('--max_bin', type=int, default=255)
ap.add_argument('--extra_feats', type=int, default=1)
ap.add_argument('--save', action='store_true')
args=ap.parse_args()
t0=time.time()
files=sorted(glob.glob(args.prepdir+'/parts/part_*.parquet'))
df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df=df[(df['fare_amount']>=2.5)&(df['orig_idx'].to_numpy()%100!=0)].reset_index(drop=True); gc.collect()
if len(df)>args.maxrows: df=df.sample(n=args.maxrows, random_state=args.seed).reset_index(drop=True)
gc.collect()
print('train loaded', df.shape, f'{time.time()-t0:.0f}s', flush=True)
hd=pd.read_parquet(P+'/holdout.parquet')
hd=hd[hd['fare_amount']>=2.5].reset_index(drop=True)
print('holdout loaded', hd.shape, flush=True)

def add_extra(d):
    d['p_air_min']=d[['p_jfk','p_lga','p_ewr']].min(axis=1).astype(np.float32)
    d['d_air_min']=d[['d_jfk','d_lga','d_ewr']].min(axis=1).astype(np.float32)
    d['air_trip']=((d['p_air_min']<2)|(d['d_air_min']<2)).astype(np.float32)
    d['jfk_manh']=(((d['p_jfk']<2)&(d['d_ctr']<8))|((d['d_jfk']<2)&(d['p_ctr']<8))).astype(np.float32)
    d['hav_per_dlon']=(d['hav']/(np.abs(d['ddlon'])+1e-3)).astype(np.float32)
    d['manh_minus_hav']=(d['manh']-d['hav']).astype(np.float32)
    return d
if args.extra_feats: df=add_extra(df); hd=add_extra(hd)

rng=np.random.default_rng(args.seed)
y=df['fare_amount'].to_numpy(np.float32); yh=hd['fare_amount'].to_numpy(np.float32)
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
pc,dc,pc3,od=codes(df); df['_pc']=pc; df['_dc']=dc; df['_pc3']=pc3; df['_od']=od
hc,hdc,hpc3,hod=codes(hd); hd['_pc']=hc; hd['_dc']=hdc; hd['_pc3']=hpc3; hd['_od']=hod
del pc,dc,pc3,od,hc,hdc,hpc3,hod; gc.collect()

if not args.no_te:
    nf=5
    fold=rng.integers(0,nf,len(df))
    prior=float(y.mean())
    def te_add(key,m):
        arr=np.full(len(df),prior,dtype=np.float32); cnt=np.zeros(len(df),dtype=np.float32)
        for f in range(nf):
            fit=fold!=f; val=fold==f
            g=df.loc[fit].groupby(key)['fare_amount']; s=g.sum(); c=g.count()
            te=((s+prior*m)/(c+m)).to_dict(); cp=c.to_dict()
            sel=np.where(val)[0]
            arr[sel]=df[key].iloc[sel].map(te).fillna(prior).to_numpy(np.float32)
            cnt[sel]=np.log1p(df[key].iloc[sel].map(cp).fillna(0).to_numpy()).astype(np.float32)
        g=df.groupby(key)['fare_amount']; s=g.sum(); c=g.count()
        te=((s+prior*m)/(c+m)).to_dict(); cp=c.to_dict()
        hd[key+'_te']=hd[key].map(te).fillna(prior).to_numpy(np.float32)
        hd[key+'_cnt']=np.log1p(hd[key].map(cp).fillna(0).to_numpy()).astype(np.float32)
        df[key+'_te']=arr; df[key+'_cnt']=cnt
    for key,m in [('_pc',20.0),('_dc',20.0),('_od',10.0),('_pc3',10.0)]:
        te_add(key,m)
        print('te',key,f'{time.time()-t0:.0f}s',flush=True)
    del fold; gc.collect()

feats=[c for c in df.columns if c not in ('fare_amount','_pc','_dc','_pc3','_od','orig_idx','pcell','dcell','pcell3')]
print('nfeat',len(feats),feats, flush=True)
Xtr=df[feats].to_numpy(np.float32); Xho=hd[feats].to_numpy(np.float32)
ho_idx=hd['orig_idx'].to_numpy(); del df; gc.collect()
print('data ready', Xtr.shape, Xho.shape, f'{time.time()-t0:.0f}s', flush=True)
params=dict(objective='regression', metric='rmse', learning_rate=args.lr,
    num_leaves=args.leaves, min_data_in_leaf=args.min_data, feature_fraction=args.ff,
    bagging_fraction=args.bf, bagging_freq=1, lambda_l2=args.l2, max_bin=args.max_bin,
    num_threads=args.threads, seed=args.seed, verbosity=-1)
dtr=lgb.Dataset(Xtr, label=y)
dho=lgb.Dataset(Xho, label=yh, reference=dtr)
model=lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dho],
    callbacks=[lgb.early_stopping(args.patience), lgb.log_evaluation(200)])
rmse=model.best_score['valid_0']['rmse']
pho=model.predict(Xho, num_iteration=model.best_iteration)
print('HOLDOUT_RMSE', rmse, 'best_iter', model.best_iteration, flush=True)
for q in [90,99,99.9,100]:
    thr=np.percentile(yh,q); m=yh<=thr
    print(f'  rmse_upto_p{q} (y<={thr:.1f}, n={m.sum()}): {np.sqrt(np.mean((pho[m]-yh[m])**2)):.4f}', flush=True)
print('  rmse excl top0.1%:', np.sqrt(np.mean((pho[yh<=np.percentile(yh,99.9)]-yh[yh<=np.percentile(yh,99.9)])**2)))
os.makedirs('models',exist_ok=True)
if args.save:
    model.save_model(f'models/lgb_{args.tag}.txt')
    np.savez(f'models/lgb_{args.tag}_holdout.npz', orig_idx=ho_idx, y=yh, pred=pho)
    np.save(f'models/lgb_{args.tag}_feats.npy', np.array(feats))
imp=sorted(zip(feats,model.feature_importance('gain')), key=lambda x:-x[1])
for f,i in imp[:15]: print(f'{f}: {i:.0f}')
print('DONE', f'{time.time()-t0:.0f}s')
