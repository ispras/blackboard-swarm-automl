import os, sys, glob, re, argparse
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool

DATA='/download_job_code/payload/sandbox/data'
S=320

def rle_decode_flatten(rle, h, w):
    m = np.zeros(h*w, np.uint8)
    if isinstance(rle, str) and rle.strip():
        s = rle.split()
        starts = np.asarray(s[0::2], dtype=np.int64) - 1
        lens = np.asarray(s[1::2], dtype=np.int64)
        for st, ln in zip(starts, lens):
            m[st:st+ln] = 1
    return m.reshape((h, w), order='C')

def find_scan_path(split, case, case_day, slice_num):
    p = os.path.join(DATA, split, case, case_day, 'scans')
    pref = f'slice_{int(slice_num):04d}_'
    for f in os.listdir(p):
        if f.startswith(pref):
            return os.path.join(p, f)
    raise FileNotFoundError((case, case_day, slice_num))

def load_norm(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(path)
    h, w = img.shape[:2]
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    if hi - lo < 1e-3:
        lo, hi = float(img.min()), float(img.max()) + 1e-3
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255.0 + 0.5).astype(np.uint8), h, w

def process_one(args):
    idx, split, slice_id, case_day, slice_num, rles = args
    case = slice_id.split('_')[0]
    path = find_scan_path(split, case, case_day, slice_num)
    img, h, w = load_norm(path)
    img_r = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
    m = np.zeros((h, w), np.uint8)
    for ci, rle in enumerate(rles):
        if isinstance(rle, str) and rle.strip() and rle == rle:
            mm = rle_decode_flatten(rle, h, w)
            m[mm > 0] = ci + 1
    m_r = cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)
    return idx, img_r.tobytes(), m_r.tobytes(), h, w

def build(split):
    if split == 'train':
        df = pd.read_csv(os.path.join(DATA,'train.csv'))
        df['case'] = df.id.str.split('_').str[0]
        df['case_day'] = df.id.str.split('_').str[:2].str.join('_')
        df['slice'] = df.id.str.extract(r'slice_(\d+)')[0].astype(int)
        index = df.drop_duplicates('id')[['id','case','case_day','slice']].sort_values('id').reset_index(drop=True)
        segmaps = {}
        for c in ['large_bowel','small_bowel','stomach']:
            sub = df[df['class'] == c]
            segmaps[c] = dict(zip(sub.id, sub.segmentation))
        recs = [(i, split, r.id, r.case_day, r.slice,
                 (segmaps['large_bowel'].get(r.id), segmaps['small_bowel'].get(r.id),
                  segmaps['stomach'].get(r.id)))
                for i, r in index.iterrows()]
    else:
        df = pd.read_csv(os.path.join(DATA,'test.csv'))
        df['case'] = df.id.str.split('_').str[0]
        df['case_day'] = df.id.str.split('_').str[:2].str.join('_')
        df['slice'] = df.id.str.extract(r'slice_(\d+)')[0].astype(int)
        index = df.drop_duplicates('id')[['id','case','case_day','slice']].sort_values('id').reset_index(drop=True)
        recs = [(i, split, r.id, r.case_day, r.slice, (None,None,None))
                for i, r in index.iterrows()]
    N = len(recs)
    root = os.path.join(DATA, f'cache_{split}_{S}')
    os.makedirs(root, exist_ok=True)
    imgs = np.lib.format.open_memmap(os.path.join(root,'images_u8.npy'), mode='w+', dtype=np.uint8, shape=(N,S,S))
    masks = np.lib.format.open_memmap(os.path.join(root,'masks_u8.npy'), mode='w+', dtype=np.uint8, shape=(N,S,S))
    meta = np.zeros((N,2), np.int32)
    with Pool(8) as pool:
        for k,(i, ib, mb, h, w) in enumerate(pool.imap_unordered(process_one, recs, chunksize=64)):
            imgs[i] = np.frombuffer(ib, np.uint8).reshape(S,S)
            masks[i] = np.frombuffer(mb, np.uint8).reshape(S,S)
            meta[i] = (h,w)
            if k % 2000 == 0:
                print(f'{split} {k}/{N}', flush=True)
    imgs.flush(); masks.flush()
    index['orig_h'] = meta[:,0]; index['orig_w'] = meta[:,1]
    index.to_csv(os.path.join(root,'index.csv'), index=False)
    print('done', split, root, N, flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='train')
    build(ap.parse_args().split)
