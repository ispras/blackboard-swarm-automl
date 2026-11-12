import os, sys, json, argparse
import numpy as np, pandas as pd, cv2
from scipy.ndimage import label as cc_label, binary_fill_holes

sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/regulus_alpha')
from metric import rle_encode

ROOT = '/download_job_code/payload/sandbox'
CACHE = f'{ROOT}/workspaces/regulus_alpha/cache'
CLASSES = ['large_bowel', 'small_bowel', 'stomach']


def cc2d(mask, min_area):
    if not mask.any():
        return mask
    lab, n = cc_label(mask)
    if n <= 1:
        return mask
    sz = np.bincount(lab.ravel()); sz[0] = 0
    keep = np.where(sz >= min_area)[0]; keep = keep[keep > 0]
    return np.isin(lab, keep)


def clean_vol(vol, min_vol, msa):
    if msa > 0:
        for z in range(vol.shape[0]):
            if vol[z].any():
                vol[z] = cc2d(vol[z], msa)
    if min_vol > 0 and vol.any():
        lab, n = cc_label(vol)
        if n > 1:
            sz = np.bincount(lab.ravel()); sz[0] = 0
            keep = np.where(sz >= min_vol)[0]; keep = keep[keep > 0]
            vol = np.isin(lab, keep)
    return vol


def origin_hw(path):
    b = os.path.basename(path).replace('.png', '').split('_')
    return int(b[3]), int(b[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', nargs='+', required=True)
    ap.add_argument('--config', default='')
    ap.add_argument('--size', type=int, default=320)
    ap.add_argument('--out', default=f'{ROOT}/final_solution/submission.csv')
    ap.add_argument('--thr', type=float, default=0.45)
    ap.add_argument('--min_area', type=int, default=0)
    ap.add_argument('--fill_holes', type=int, default=0)
    ap.add_argument('--min3d', type=int, default=0)
    a = ap.parse_args()

    files, weights = [], []
    for p in a.probs:
        parts = p.rsplit(':', 1)
        if len(parts) == 2 and os.path.exists(parts[0]):
            files.append(parts[0]); weights.append(float(parts[1]))
        else:
            files.append(p); weights.append(1.0)
    wsum = sum(weights)
    arrs = [np.load(f, mmap_mode='r') for f in files]
    N = arrs[0].shape[0]
    for ar in arrs:
        assert ar.shape[0] == N

    cfg = None
    if a.config:
        cfg = json.load(open(a.config))['per_class']
        print('config', json.dumps(cfg), flush=True)

    meta = pd.read_csv(f'{CACHE}/test_meta_{a.size}.csv')
    assert len(meta) == N
    tdf = pd.read_csv(f'{ROOT}/data/test.csv')
    rec = {}
    day_ids = meta['day_id'].values
    for d in np.unique(day_ids):
        idxs = np.where(day_ids == d)[0]
        idxs = idxs[np.argsort(meta['pos'].values[idxs])]
        chunk = np.zeros((len(idxs), 3, a.size, a.size), dtype=np.float32)
        for ar, w in zip(arrs, weights):
            chunk += ar[idxs].astype(np.float32) * w
        chunk /= wsum
        pred = np.zeros((len(idxs), 3, a.size, a.size), dtype=bool)
        for c in range(3):
            if cfg:
                cc = cfg[CLASSES[c]]
                thr = float(cc['thr']); dil = int(cc.get('plane_dil', cc.get('dil', 0)))
                mv = int(cc.get('min_vol', 0)); msa = int(cc.get('msa', cc.get('min_slice_area', 0)))
            else:
                thr = a.thr; dil = 0; mv = a.min3d; msa = a.min_area
            vol = (chunk[:, c] > thr * 255.0).astype(np.uint8)
            if dil > 0:
                k = np.ones((3, 3), np.uint8)
                for z in range(vol.shape[0]):
                    if vol[z].any():
                        vol[z] = cv2.dilate(vol[z], k, iterations=dil)
            vb = vol.astype(bool)
            if mv > 0 or msa > 0:
                vb = clean_vol(vb, mv, msa)
            pred[:, c] = vb
        if a.fill_holes:
            for j in range(len(idxs)):
                for c in range(3):
                    if pred[j, c].any():
                        pred[j, c] = binary_fill_holes(pred[j, c])
        for j, i in enumerate(idxs):
            sid = meta['id'].values[i]
            h, w = origin_hw(meta['path'].values[i])
            for c in range(3):
                m = pred[j, c]
                if m.any():
                    mo = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    rec[(sid, CLASSES[c])] = rle_encode(mo)
                else:
                    rec[(sid, CLASSES[c])] = ''
    out = tdf.copy()
    out['predicted'] = [rec.get((i, c), '') for i, c in zip(out['id'], out['class'])]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print('wrote', a.out, out.shape, 'nonempty', round((out['predicted'].str.len() > 0).mean(), 4), flush=True)


if __name__ == '__main__':
    main()
