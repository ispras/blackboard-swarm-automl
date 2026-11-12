import glob, time
import numpy as np
import pyarrow.parquet as pq

V2 = sorted(glob.glob('/download_job_code/payload/sandbox/workspaces/mimosa_omicron/feat_v2/*.parquet'))
NPZ = '/download_job_code/payload/sandbox/workspaces/mimosa_omicron/te_maps.npz'
OUT = '/download_job_code/payload/sandbox/workspaces/aldebaran_omicron/extra_aligned.npy'
N = 55_413_942
COLS = ['te_pcell','te_dcell','te_route','p_cell_count','d_cell_count','route_count']

te = np.load(NPZ)
te_p = te['te_pcell_oof']; te_d = te['te_dcell_oof']; te_r = te['te_route_oof']
print('te loaded', te_p.shape, flush=True)

arr = np.full((N, len(COLS)), np.nan, dtype=np.float32)
off = 0
t0 = time.time()
for i, p in enumerate(V2):
    t = pq.read_table(p, columns=['orig_idx','p_cell_count','d_cell_count','route_count'])
    oi = t['orig_idx'].to_numpy().astype(np.int64)
    n = len(oi)
    sl = slice(off, off+n)
    arr[oi, 0] = te_p[sl]
    arr[oi, 1] = te_d[sl]
    arr[oi, 2] = te_r[sl]
    arr[oi, 3] = t['p_cell_count'].to_numpy().astype(np.float32)
    arr[oi, 4] = t['d_cell_count'].to_numpy().astype(np.float32)
    arr[oi, 5] = t['route_count'].to_numpy().astype(np.float32)
    off += n
    del t, oi
    if i % 3 == 0:
        print(f'part {i} off {off} {time.time()-t0:.0f}s', flush=True)
np.save(OUT, arr)
print('saved', OUT, arr.shape, 'filled frac', np.isfinite(arr[:,0]).mean(), time.time()-t0, flush=True)
