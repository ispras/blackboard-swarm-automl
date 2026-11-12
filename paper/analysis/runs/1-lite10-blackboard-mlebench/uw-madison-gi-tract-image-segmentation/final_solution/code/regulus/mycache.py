import numpy as np, pandas as pd

ROOT = '/download_job_code/payload/sandbox'
CACHE = f'{ROOT}/workspaces/regulus_alpha/cache'


def load_val(ids_file, size=320):
    """Return (vi DataFrame, gt combined labels (N,H,W) uint8, groups list of index arrays).
    Uses regulus_alpha corrected (order='C') mask cache. groups = case-day groups."""
    meta = pd.read_csv(f'{CACHE}/train_meta_{size}.csv')
    pos = {v: i for i, v in enumerate(meta['id'].values)}
    ids = pd.read_csv(ids_file)['id'].values
    rows = np.array([pos[v] for v in ids])
    vi = meta.iloc[rows].reset_index(drop=True)
    msk = np.memmap(f'{CACHE}/train_msk_{size}.dat', dtype=np.uint8, mode='r',
                    shape=(len(meta), 3, size, size))
    m = np.asarray(msk[rows]) > 0
    gt = (m[:, 0].astype(np.uint8)
          + 2 * m[:, 1].astype(np.uint8)
          + 3 * m[:, 2].astype(np.uint8))
    groups_all = [g.index.values for _, g in vi.groupby('day_key', sort=False)]
    return vi, gt, groups_all
