import numpy as np, json, sys, pyarrow.parquet as pq, xgboost as xgb, os
W='/download_job_code/payload/sandbox/workspaces/antares_omicron'
tag=sys.argv[1]
lo,hi=(float(sys.argv[2]),float(sys.argv[3])) if len(sys.argv)>3 else (2.5,200.0)
meta=json.load(open(f'{W}/{tag}_meta.json')); FEATS=meta['feats']
TE=[f for f in FEATS if f.startswith('te_') or f.startswith('cnt_') or f=='same_cell05']
base=[f for f in FEATS if f not in TE]
te=pq.read_table(f'{W}/feat/test_feats.parquet')
X=np.empty((te.num_rows,len(FEATS)),dtype=np.float32)
for i,f in enumerate(base): X[:,i]=te[f].to_numpy(zero_copy_only=False)
if TE:
    tt=pq.read_table(f'{W}/feat/te_test.parquet')
    for j,f in enumerate(TE): X[:,len(base)+j]=tt[f].to_numpy(zero_copy_only=False)
bst=xgb.Booster(); bst.load_model(f'{W}/{tag}.json')
pt=bst.predict(xgb.DMatrix(X,feature_names=FEATS,missing=np.nan),iteration_range=(0,int(meta['best_iter'])+1))
pt=np.clip(pt,lo,hi)
np.save(f'{W}/{tag}_testpred.npy',pt.astype(np.float32))
hp=te['has_pickup'].to_numpy(); hd=te['has_dropoff'].to_numpy(); miss=(hp<1)|(hd<1)
print(tag,'test mean',pt.mean(),'missing mean',pt[miss].mean(),'missing max',pt[miss].max(),'overall max',pt.max())
