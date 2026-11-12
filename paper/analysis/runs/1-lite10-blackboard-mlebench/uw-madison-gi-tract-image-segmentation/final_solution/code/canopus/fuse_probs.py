import argparse, os, sys
import numpy as np, pandas as pd

def align_maps(model_ids, canon_ids):
    pos = {v: i for i, v in enumerate(model_ids)}
    return np.array([pos.get(v, -1) for v in canon_ids], np.int64)

def fuse(inputs, weights, canon_ids, out_path, chunk=512):
    maps = []
    for npy, idcsv in inputs:
        mid = pd.read_csv(idcsv)['id'].values
        maps.append((npy, align_maps(mid, canon_ids)))
    n = len(canon_ids)
    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.uint8, shape=(n,3,320,320))
    W = float(sum(weights))
    for s in range(0, n, chunk):
        e = min(n, s+chunk)
        acc = np.zeros((e-s,3,320,320), np.float32)
        for (npy, m), w in zip(maps, weights):
            rows = m[s:e]
            p = np.load(npy, mmap_mode='r')
            # missing ids -> 0 (should not happen)
            valid = rows >= 0
            acc[valid] += w * np.asarray(p[rows[valid]], np.float32)
        acc /= W
        out[s:e] = np.clip(acc, 0, 255).round().astype(np.uint8)
        print('chunk', s, e, flush=True)
    out.flush()
    pd.DataFrame({'id': canon_ids}).to_csv(out_path.replace('.npy','_ids.csv'), index=False)
    print('wrote', out_path, n, flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--canon_ids', required=True, help='csv with id column, canonical order')
    ap.add_argument('--out', required=True)
    ap.add_argument('--probs', nargs='+', required=True)
    ap.add_argument('--ids', nargs='+', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=None)
    a = ap.parse_args()
    canon = pd.read_csv(a.canon_ids)['id'].values
    weights = a.weights or [1.0]*len(a.probs)
    assert len(weights) == len(a.probs) == len(a.ids)
    fuse(list(zip(a.probs, a.ids)), weights, canon, a.out)
