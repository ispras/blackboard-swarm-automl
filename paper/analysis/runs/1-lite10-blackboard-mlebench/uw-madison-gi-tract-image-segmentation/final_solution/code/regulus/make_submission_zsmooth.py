import os, sys, argparse
import numpy as np, pandas as pd, cv2
from scipy.ndimage import gaussian_filter1d

ROOT = '/download_job_code/payload/sandbox'
sys.path.insert(0, f'{ROOT}/workspaces/regulus_alpha')
from metric import rle_encode

CACHE_TE = f'{ROOT}/data/cache_test_320'
CLASSES = ['large_bowel', 'small_bowel', 'stomach']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', required=True, help='fused test probs in canopus cache_test_320 index order')
    ap.add_argument('--sigma', type=float, default=1.5)
    ap.add_argument('--thrs', default='0.50,0.45,0.40')
    ap.add_argument('--out', default=f'{ROOT}/workspaces/regulus_alpha/probs/zs_submission.csv')
    a = ap.parse_args()
    thrs = [float(x) for x in a.thrs.split(',')]
    index = pd.read_csv(f'{CACHE_TE}/index.csv')
    P = np.load(a.probs, mmap_mode='r')
    assert len(P) == len(index), (len(P), len(index))
    sub_rows = []
    n_day = 0
    for cd, g in index.groupby('case_day', sort=False):
        ii = g.index.values
        order = ii[np.argsort(g['slice'].values)]
        chunk = np.zeros((len(order), 3, 320, 320), np.float32)
        for c in range(3):
            chunk[:, c] = gaussian_filter1d(np.asarray(P[order, c], dtype=np.float32),
                                            a.sigma, axis=0, mode='nearest')
        for j, i in enumerate(order):
            r = index.iloc[i]
            oh, ow = int(r.orig_h), int(r.orig_w)
            for c, cls in enumerate(CLASSES):
                m = chunk[j, c] > thrs[c] * 255.0
                if m.any():
                    mo = cv2.resize(m.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
                    sub_rows.append((r.id, cls, rle_encode(mo)))
                else:
                    sub_rows.append((r.id, cls, ''))
        n_day += 1
        if n_day % 20 == 0:
            print('day', n_day, flush=True)
    sub = pd.DataFrame(sub_rows, columns=['id', 'class', 'predicted'])
    te = pd.read_csv(f'{ROOT}/data/test.csv')
    sub = te[['id', 'class']].merge(sub, on=['id', 'class'], how='left')
    sub['predicted'] = sub['predicted'].fillna('')
    sub.to_csv(a.out, index=False)
    print('WROTE', a.out, 'nonempty', int((sub.predicted != '').sum()), 'sigma', a.sigma, flush=True)


if __name__ == '__main__':
    main()
