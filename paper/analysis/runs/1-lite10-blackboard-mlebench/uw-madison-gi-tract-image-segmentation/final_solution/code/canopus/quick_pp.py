import sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from metric import hd_score, points_of, CLASSES
from postproc3d import clean_volume
from scipy import ndimage

CACHE = "/download_job_code/payload/sandbox/data/cache_train_320"
ap = argparse.ArgumentParser(); ap.add_argument('--probs', required=True); ap.add_argument('--ids', required=True); ap.add_argument('--thr', type=float, default=0.35)
a = ap.parse_args()
index = pd.read_csv(f'{CACHE}/index.csv'); mp = {v:i for i,v in enumerate(index['id'])}
ids = pd.read_csv(a.ids)['id'].values; rows = np.array([mp[v] for v in ids])
vi = index.iloc[rows].reset_index(drop=True); gt = np.asarray(np.load(f'{CACHE}/masks_u8.npy', mmap_mode='r')[rows])
P = np.asarray(np.load(a.probs, mmap_mode='r'))
groups = [(cd, g.index.values) for cd, g in vi.groupby('case_day')]
pred0 = P > int(a.thr*255)

def score(pred):
    out = {}
    for c in range(3):
        g = (gt == c+1); p = pred[:, c]
        inter = np.logical_and(p, g).reshape(len(p),-1).sum(1); s = p.reshape(len(p),-1).sum(1)+g.reshape(len(g),-1).sum(1)
        d = np.where(s>0, 2*inter/np.maximum(s,1), 1.0).mean()
        hds = [hd_score(points_of(p[ii]), points_of(g[ii])) for _, ii in groups]
        out[c] = (float(d), float(np.mean(hds)))
    dice = np.mean([out[c][0] for c in range(3)]); hd = np.mean([out[c][1] for c in range(3)])
    return dice, hd, 0.4*dice+0.6*hd

def apply(fn):
    pred = pred0.copy()
    for cd, ii in groups:
        for c in range(3):
            pred[ii, c] = fn(pred[ii, c])
    return pred

def rm_isolated(vol, min_area):
    out = vol.copy()
    for z in range(out.shape[0]):
        if out[z].sum() == 0: continue
        prev = out[z-1].sum() if z > 0 else 0
        nxt = out[z+1].sum() if z+1 < out.shape[0] else 0
        if out[z].sum() < min_area and prev == 0 and nxt == 0:
            out[z] = False
    return out

configs = {
 'none': lambda v: v,
 'rm_iso10': lambda v: rm_isolated(v, 10),
 'rm_iso30': lambda v: rm_isolated(v, 30),
 'minvol100': lambda v: clean_volume(v, min_vol=100),
 'minvol400': lambda v: clean_volume(v, min_vol=400),
 'minvol1000': lambda v: clean_volume(v, min_vol=1000),
 'msa20': lambda v: clean_volume(v, min_slice_area=20),
 'msa50': lambda v: clean_volume(v, min_slice_area=50),
}
res = {}
for name, fn in configs.items():
    d, h, combo = score(apply(fn))
    res[name] = (d, h, combo)
    print(f'{name:12s} dice={d:.4f} hd={h:.4f} combo={combo:.4f}', flush=True)
json.dump(res, open('/tmp/opencode/quick_pp_B.json','w'), indent=1)
