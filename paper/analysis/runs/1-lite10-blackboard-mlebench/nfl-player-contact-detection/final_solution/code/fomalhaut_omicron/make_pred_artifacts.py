"""Convert oof_<feat>.npy / ptest_<feat>.npy to standard contact_id-keyed parquet."""
import pandas as pd, numpy as np, sys

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--feat', default='feat3')   # data parquet name
ap.add_argument('--pred', default=None)      # prediction npy suffix (defaults to feat)
ap.add_argument('--out', default=None)       # output suffix (defaults to feat)
args = ap.parse_args()
feat = args.feat
pred = args.pred or feat
out = args.out or feat
B = 'workspaces/fomalhaut_omicron/cache/'

d = pd.read_parquet(B + '%s_train.parquet' % feat, columns=['contact_id', 'game_play', 'p1', 'p2', 'step'])
d = d.sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
oof = np.load(B + 'oof_%s.npy' % pred)
assert len(d) == len(oof)
pd.DataFrame({'contact_id': d.contact_id.values, 'prob': oof.astype(np.float32)}).to_parquet(
    'workspaces/fomalhaut_omicron/oof_pred_%s.parquet' % out, index=False)

csv = pd.read_csv('data/sample_submission.csv', usecols=['contact_id'])
pt = np.load(B + 'ptest_%s.npy' % pred)
ids = np.load(B + 'ptest_ids_%s.npy' % pred, allow_pickle=True)
p = pd.DataFrame({'contact_id': ids, 'prob': pt.astype(np.float32)})
p = csv.merge(p, on='contact_id', how='left')
assert p.prob.notna().all()
p.to_parquet('workspaces/fomalhaut_omicron/test_pred_%s.parquet' % out, index=False)
print('saved oof/test pred parquet for', out, len(d), len(p))
