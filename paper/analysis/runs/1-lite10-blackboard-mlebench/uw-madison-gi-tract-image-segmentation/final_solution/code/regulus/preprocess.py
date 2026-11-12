import os, sys, glob, json, argparse
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool

cv2.setNumThreads(1)
SIZE = 320
DATA = '/download_job_code/payload/sandbox/data'
OUT = '/download_job_code/payload/sandbox/workspaces/regulus_alpha/cache'


def rle_decode(seg, shape):
    h, w = shape
    out = np.zeros(h * w, dtype=np.uint8)
    if isinstance(seg, str) and len(seg) > 0:
        s = seg.split()
        starts = np.asarray(s[0::2], dtype=np.int64) - 1
        lengths = np.asarray(s[1::2], dtype=np.int64)
        for st, ln in zip(starts, lengths):
            out[st:st + ln] = 1
    return out.reshape((h, w), order='C')


def norm_img(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 1.0), np.percentile(img, 99.0)
    if hi <= lo:
        hi = img.max() if img.max() > lo else lo + 1.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo) * 255.0
    return img.astype(np.uint8)


def proc(args):
    idx, path, segs, size = args
    base = os.path.basename(path)
    parts = base.replace('.png', '').split('_')
    w, h = int(parts[2]), int(parts[3])
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError('cannot read ' + path)
    im = norm_img(im)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    masks = np.zeros((3, size, size), dtype=np.uint8)
    for ci in range(3):
        m = rle_decode(segs[ci], (h, w))
        if m.sum() > 0:
            masks[ci] = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return idx, im, masks


def build_split(kind, size, workers):
    os.makedirs(OUT, exist_ok=True)
    if kind == 'train':
        df = pd.read_csv(f'{DATA}/train.csv')
        # pivot masks
        piv = df.pivot_table(index='id', columns='class', values='segmentation', aggfunc='first')
        ids = list(dict.fromkeys(df['id']))
        piv = piv.reindex(ids)
        paths = {}
        for p in glob.glob(f'{DATA}/train/*/*/scans/*.png'):
            case_day = os.path.basename(os.path.dirname(os.path.dirname(p)))
            slc = os.path.basename(p).split('_')[1]
            paths[f'{case_day}_slice_{slc}'] = p
        rows = []
        for sid in ids:
            case, day, slc = sid.split('_')[0], sid.split('_')[1], int(sid.split('_')[3])
            rows.append((sid, case, day, slc, paths[sid]))
        meta = pd.DataFrame(rows, columns=['id', 'case', 'day', 'slice', 'path'])
        meta = meta.sort_values(['case', 'day', 'slice']).reset_index(drop=True)
        segs_all = []
        for sid in meta['id']:
            r = piv.loc[sid]
            segs_all.append([r.get('large_bowel', ''), r.get('small_bowel', ''), r.get('stomach', '')])
    else:
        df = pd.read_csv(f'{DATA}/test.csv')
        ids = list(dict.fromkeys(df['id']))
        paths = {}
        for p in glob.glob(f'{DATA}/test/*/*/scans/*.png'):
            case_day = os.path.basename(os.path.dirname(os.path.dirname(p)))
            slc = os.path.basename(p).split('_')[1]
            paths[f'{case_day}_slice_{slc}'] = p
        rows = []
        for sid in ids:
            case, day, slc = sid.split('_')[0], sid.split('_')[1], int(sid.split('_')[3])
            rows.append((sid, case, day, slc, paths[sid]))
        meta = pd.DataFrame(rows, columns=['id', 'case', 'day', 'slice', 'path'])
        meta = meta.sort_values(['case', 'day', 'slice']).reset_index(drop=True)
        segs_all = [['', '', ''] for _ in range(len(meta))]

    N = len(meta)
    img_mm = np.memmap(f'{OUT}/{kind}_img_{size}.dat', dtype=np.uint8, mode='w+', shape=(N, size, size))
    msk_mm = np.memmap(f'{OUT}/{kind}_msk_{size}.dat', dtype=np.uint8, mode='w+', shape=(N, 3, size, size))
    tasks = [(i, meta.at[i, 'path'], segs_all[i], size) for i in range(N)]
    with Pool(workers) as pool:
        for k, (idx, im, masks) in enumerate(pool.imap_unordered(proc, tasks, chunksize=16)):
            img_mm[idx] = im
            msk_mm[idx] = masks
            if k % 2000 == 0:
                print(k, '/', N, flush=True)
    img_mm.flush(); msk_mm.flush(); del img_mm, msk_mm

    # day ids & position
    meta['day_key'] = meta['case'] + '_' + meta['day']
    codes = {k: i for i, k in enumerate(dict.fromkeys(meta['day_key']))}
    meta['day_id'] = meta['day_key'].map(codes).astype(np.int32)
    meta['pos'] = meta.groupby('day_id').cumcount().astype(np.int32)
    meta['day_n'] = meta.groupby('day_id')['id'].transform('size').astype(np.int32)
    meta.to_csv(f'{OUT}/{kind}_meta_{size}.csv', index=False)
    print('saved', kind, N, 'SIZE', size)
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='train')
    ap.add_argument('--size', type=int, default=320)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    meta = build_split(a.kind, a.size, a.workers)
    # compute has_ flags from masks (reopen memmap)
    msk = np.memmap(f'{OUT}/{a.kind}_msk_{a.size}.dat', dtype=np.uint8, mode='r', shape=(len(meta), 3, a.size, a.size))
    meta['has_lb'] = (msk[:, 0].reshape(len(meta), -1).max(1) > 0).astype(int)
    meta['has_sb'] = (msk[:, 1].reshape(len(meta), -1).max(1) > 0).astype(int)
    meta['has_st'] = (msk[:, 2].reshape(len(meta), -1).max(1) > 0).astype(int)
    meta.to_csv(f'{OUT}/{a.kind}_meta_{a.size}.csv', index=False)
    print(meta.head())
    print(meta[['has_lb', 'has_sb', 'has_st']].mean())
