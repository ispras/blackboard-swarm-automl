import os, sys, argparse, json
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postproc3d import clean_volume

ROOT = "/download_job_code/payload/sandbox"
CACHE_TE = f"{ROOT}/data/cache_test_320"
CLASSES = ["large_bowel", "small_bowel", "stomach"]

def rle_encode(mask, order='C'):
    p = np.asarray(mask).flatten(order=order)
    p = np.concatenate([[0], p, [0]])
    runs = np.where(p[1:] != p[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', nargs='+', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=None)
    ap.add_argument('--thr', nargs='+', type=float, default=[0.5, 0.5, 0.5])
    ap.add_argument('--min_vol', nargs='+', type=int, default=[0, 0, 0])
    ap.add_argument('--keep_top', nargs='+', type=int, default=[0, 0, 0])
    ap.add_argument('--min_slice_area', nargs='+', type=int, default=[0, 0, 0])
    ap.add_argument('--config', default=None, help='json from tune_all (overrides thr/min_vol)')
    ap.add_argument('--rle_order', default='C', choices=['C','F'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.config:
        cfg = json.load(open(a.config))['per_class']
        a.thr = [cfg[str(c)]['thr'] for c in range(3)]
        a.min_vol = [cfg[str(c)]['min_vol'] for c in range(3)]
        a.min_slice_area = [cfg[str(c)].get('msa', 0) for c in range(3)]
        a.keep_top = [cfg[str(c)].get('keep_top', 0) for c in range(3)]
    index = pd.read_csv(f'{CACHE_TE}/index.csv')
    assert index['id'].is_unique
    P = None
    W = a.weights or [1.0] * len(a.probs)
    for f, w in zip(a.probs, W):
        p = np.load(f, mmap_mode='r')
        assert len(p) == len(index), (f, len(p), len(index))
        P = np.asarray(p, np.float32) * w if P is None else P + np.asarray(p, np.float32) * w
    P /= sum(W)
    pred = np.zeros((len(index), 3, 320, 320), bool)
    for c in range(3):
        pred[:, c] = P[:, c] > int(a.thr[c] * 255)
    # 3D cleanup per case-day per class
    for cd, g in index.groupby('case_day'):
        ii = g.index.values
        for c in range(3):
            if a.min_vol[c] or a.min_slice_area[c]:
                pred[ii, c] = clean_volume(pred[ii, c], a.min_vol[c], a.keep_top[c], a.min_slice_area[c])
    rows = []
    for i in range(len(index)):
        r = index.iloc[i]
        oh, ow = int(r.orig_h), int(r.orig_w)
        for c, cls in enumerate(CLASSES):
            m = pred[i, c]
            if m.sum() == 0:
                rows.append((r.id, cls, '')); continue
            mr = cv2.resize(m.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
            rows.append((r.id, cls, rle_encode(mr, a.rle_order)))
        if i % 2000 == 0:
            print(i, len(index), flush=True)
    sub = pd.DataFrame(rows, columns=['id', 'class', 'predicted'])
    te = pd.read_csv(f'{ROOT}/data/test.csv')
    sub = te[['id', 'class']].merge(sub, on=['id', 'class'], how='left')
    sub['predicted'] = sub['predicted'].fillna('')
    sub.to_csv(a.out, index=False)
    print('wrote', a.out, len(sub), 'nonempty', (sub.predicted != '').sum(), flush=True)

if __name__ == '__main__':
    main()
