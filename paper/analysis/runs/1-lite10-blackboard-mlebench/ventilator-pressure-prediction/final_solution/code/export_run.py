import sys, os
import numpy as np
import pandas as pd

CACHE = 'workspaces/sirius_tau/cache'
PREDS = 'workspaces/sirius_tau/preds'
os.makedirs(PREDS, exist_ok=True)

run = sys.argv[1]
name = sys.argv[2]
val_idx = np.load(f'{CACHE}/val_idx.npy')
ids = np.load(f'{CACHE}/id_train.npy')
vid = ids[val_idx].reshape(-1)
vp = np.load(f'{run}/val_pred.npy').astype(np.float64).reshape(-1)
assert len(vid) == len(vp), (len(vid), len(vp))
pd.DataFrame({'id': vid, 'pressure': vp}).to_csv(f'{PREDS}/val_seed42_{name}.csv', index=False)
t = pd.read_csv(f'{run}/test_pred.csv')
t.to_csv(f'{PREDS}/test_{name}.csv', index=False)
print('exported', name, 'val rows', len(vid), 'test rows', len(t))
