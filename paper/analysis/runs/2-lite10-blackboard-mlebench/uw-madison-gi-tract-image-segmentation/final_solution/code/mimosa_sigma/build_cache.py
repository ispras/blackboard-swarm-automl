import os, sys, json, glob, re, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import GroupKFold

ROOT = '/download_job_code/payload/sandbox'
OUT = f'{ROOT}/workspaces/mimosa_sigma/cache'
os.makedirs(OUT, exist_ok=True)
SZ = 320
cv2.setNumThreads(1)

def rle_decode(rle, shape):
    h, w = shape
    img = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle, str) and len(rle.strip()):
        s = np.asarray(rle.split(), dtype=np.int64)
        starts = s[0::2] - 1
        lengths = s[1::2]
        for st, ln in zip(starts, lengths):
            img[st:st + ln] = 1
    return img.reshape((h, w))  # C-order verified

def build_path_map(split):
    m = {}
    for p in glob.glob(f'{ROOT}/data/{split}/*/*/scans/*.png'):
        case_day = os.path.basename(os.path.dirname(os.path.dirname(p)))
        bn = os.path.basename(p)
        sl = bn.split('_')[1]
        m[f'{case_day}_slice_{sl}'] = p
    return m

def load_one(args):
    idx, path = args
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    if p99 <= p1:
        p1, p99 = float(img.min()), float(img.max()) + 1e-6
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1) * 255.0
    img = img.astype(np.uint8)
    img = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_AREA)
    return idx, img, h, w

def process_split(split):
    pm = build_path_map(split)
    ids = sorted(pm.keys())
    N = len(ids)
    print(split, 'N', N, flush=True)
    img_mm = np.lib.format.open_memmap(f'{OUT}/{split}_img.npy', mode='w+', dtype=np.uint8, shape=(N, SZ, SZ))
    hw = np.zeros((N, 2), dtype=np.int32)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, (idx, img, h, w) in enumerate(ex.map(load_one, list(enumerate(map(pm.get, ids))))):
            img_mm[idx] = img
            hw[idx] = (h, w)
            if (k + 1) % 5000 == 0:
                print(split, k + 1, f'{time.time()-t0:.0f}s', flush=True)
    img_mm.flush()
    np.save(f'{OUT}/{split}_hw.npy', hw)
    # prev/next within same study
    study = [i.rsplit('_slice_', 1)[0] for i in ids]
    prev = np.arange(N); nxt = np.arange(N)
    for i in range(1, N):
        if study[i] == study[i - 1]:
            prev[i] = i - 1
    for i in range(N - 1):
        if study[i] == study[i + 1]:
            nxt[i] = i + 1
    np.save(f'{OUT}/{split}_prev.npy', prev)
    np.save(f'{OUT}/{split}_next.npy', nxt)
    with open(f'{OUT}/{split}_ids.json', 'w') as f:
        json.dump(ids, f)
    print(split, 'done', f'{time.time()-t0:.0f}s', flush=True)
    return ids, hw, pm

# ---- images
tr_ids, tr_hw, tr_pm = process_split('train')
te_ids, te_hw, te_pm = process_split('test')

# ---- masks
tr = pd.read_csv(f'{ROOT}/data/train.csv')
classes = ['large_bowel', 'small_bowel', 'stomach']
id2i = {x: i for i, x in enumerate(tr_ids)}
N = len(tr_ids)
mask_mm = np.lib.format.open_memmap(f'{OUT}/train_mask.npy', mode='w+', dtype=np.uint8, shape=(N, 3, SZ, SZ))
print('decoding masks...', flush=True)
t0 = time.time()
cnt = 0
tr = tr.rename(columns={'class':'cls'})
for row in tr.itertuples():
    if not isinstance(row.segmentation, str) or not len(row.segmentation.strip()):
        continue
    i = id2i.get(row.id)
    if i is None:
        continue
    c = classes.index(row.cls)
    h, w = tr_hw[i]
    m = rle_decode(row.segmentation, (int(h), int(w)))
    m = cv2.resize(m, (SZ, SZ), interpolation=cv2.INTER_NEAREST)
    mask_mm[i, c] = m
    cnt += 1
mask_mm.flush()
print('masks', cnt, f'{time.time()-t0:.0f}s', flush=True)

# ---- folds
tr['case'] = tr.id.str.split('_').str[0]
cases = tr[['case']].drop_duplicates().reset_index(drop=True)
gkf = GroupKFold(n_splits=5)
case_fold = {}
for f, (_, va) in enumerate(gkf.split(tr, groups=tr['case'])):
    for c in tr.iloc[va]['case'].unique():
        case_fold[c] = f
cases['fold'] = cases['case'].map(case_fold)
cases.to_csv(f'{OUT}/folds.csv', index=False)
print(cases.fold.value_counts().sort_index().to_dict())
print('ALL DONE')
