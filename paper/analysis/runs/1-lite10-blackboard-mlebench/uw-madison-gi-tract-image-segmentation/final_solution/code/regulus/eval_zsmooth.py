import os, sys, argparse, json
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOT = '/download_job_code/payload/sandbox'
sys.path.insert(0, f'{ROOT}/workspaces/canopus_phi')
from metric import points_of, CLASSES  # noqa

CACHE = f'{ROOT}/data/cache_train_320'
MAXPTS = 2000


def hd_score(a_pts, b_pts):
    if len(a_pts) == 0 and len(b_pts) == 0:
        return 1.0
    if len(a_pts) == 0 or len(b_pts) == 0:
        return 0.0
    if len(a_pts) > MAXPTS:
        a_pts = a_pts[np.linspace(0, len(a_pts) - 1, MAXPTS).astype(int)]
    if len(b_pts) > MAXPTS:
        b_pts = b_pts[np.linspace(0, len(b_pts) - 1, MAXPTS).astype(int)]
    ta, tb = cKDTree(a_pts), cKDTree(b_pts)
    da, _ = tb.query(a_pts); db, _ = ta.query(b_pts)
    return float(np.clip(1.0 - max(da.max(), db.max()), 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--thrs', default='0.50,0.45,0.40')
    ap.add_argument('--sigmas', default='0,0.75,1.0,1.25,1.5,2.0')
    a = ap.parse_args()
    ids = pd.read_csv(a.ids)['id'].values
    index = pd.read_csv(f'{CACHE}/index.csv')
    pos = {v: i for i, v in enumerate(index['id'].values)}
    rows = np.array([pos[v] for v in ids])
    vi = index.iloc[rows].reset_index(drop=True)
    gt = np.asarray(np.load(f'{CACHE}/masks_u8.npy', mmap_mode='r')[rows])
    P = np.load(a.probs, mmap_mode='r')
    groups = [g.index.values for _, g in vi.groupby('case_day', sort=False)]
    thrs = [float(x) for x in a.thrs.split(',')]
    sigmas = [float(x) for x in a.sigmas.split(',')]
    # precompute gt points
    gt_pts = [[points_of(gt[ii] == c + 1) for ii in groups] for c in range(3)]
    for sigma in sigmas:
        comps = []
        out = []
        for c in range(3):
            Pc = np.asarray(P[:, c], dtype=np.float32)
            if sigma > 0:
                for ii in groups:
                    Pc[ii] = gaussian_filter1d(Pc[ii], sigma, axis=0, mode='nearest')
            pv = Pc > thrs[c] * 255
            g = (gt == c + 1)
            inter = np.logical_and(pv, g).reshape(len(pv), -1).sum(1)
            s = pv.reshape(len(pv), -1).sum(1) + g.reshape(len(g), -1).sum(1)
            d = float(np.where(s > 0, 2 * inter / np.maximum(s, 1), 1.0).mean())
            hs = [hd_score(gt_pts[c][k], points_of(pv[ii])) for k, ii in enumerate(groups)]
            h = float(np.mean(hs))
            combo = 0.4 * d + 0.6 * h
            out.append((d, h, combo)); comps.append(combo)
        print(f'sigma {sigma}: dice {np.mean([o[0] for o in out]):.4f} hd {np.mean([o[1] for o in out]):.4f} '
              f'combo {np.mean(comps):.4f} per_cls {[(round(x[0],4),round(x[1],4)) for x in out]}', flush=True)


if __name__ == '__main__':
    main()
