import os, json, time
import numpy as np, pandas as pd, cv2
ROOT = '/download_job_code/payload/sandbox'
C = f'{ROOT}/workspaces/mimosa_sigma/cache'
SZ = 320
cv2.setNumThreads(4)

def rle_decode_c(rle, h, w):
    img = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        s = np.asarray(rle.split(), dtype=np.int64)
        for st, ln in zip(s[0::2] - 1, s[1::2]):
            img[st:st + ln] = 1
    return img.reshape((h, w))  # C-order verified

ids = json.load(open(f'{C}/train_ids.json'))
id2i = {x: i for i, x in enumerate(ids)}
hw = np.load(f'{C}/train_hw.npy')
mask = np.lib.format.open_memmap(f'{C}/train_mask.npy', mode='w+', dtype=np.uint8, shape=(len(ids), 3, SZ, SZ))
tr = pd.read_csv(f'{ROOT}/data/train.csv', dtype=str).rename(columns={'class': 'cls'})
classes = ['large_bowel', 'small_bowel', 'stomach']
t0 = time.time(); cnt = 0
for r in tr.itertuples():
    if not isinstance(r.segmentation, str) or not r.segmentation.strip():
        continue
    i = id2i[r.id]; h, w = int(hw[i][0]), int(hw[i][1])
    m = rle_decode_c(r.segmentation, h, w)
    m = cv2.resize(m, (SZ, SZ), interpolation=cv2.INTER_NEAREST)
    mask[i, classes.index(r.cls)] = m
    cnt += 1
mask.flush()
print('rebuilt masks', cnt, 'nonzero slices', int((mask.reshape(len(ids), -1).any(1)).sum()), f'{time.time()-t0:.0f}s')
