import os, re, glob, sys
import numpy as np, pandas as pd, cv2

ROOT = "/download_job_code/payload/sandbox"
WS = f"{ROOT}/workspaces/aldebaran_omicron"
SIZE = 320
OUT = f"{WS}/cache"
os.makedirs(OUT, exist_ok=True)

split = sys.argv[1] if len(sys.argv) > 1 else "train"

def img_path(split, case, cday, sl):
    base = f"{ROOT}/data/{split}/{case}/{cday}/scans"
    g = glob.glob(f"{base}/slice_{sl}_*.png")
    if not g:
        raise FileNotFoundError(f"{base}/slice_{sl}_*.png")
    return g[0]

def norm_u8(im):
    lo, hi = np.percentile(im, 1), np.percentile(im, 99)
    if hi <= lo:
        hi = float(im.max()) if im.max() > lo else lo + 1
    im = np.clip(im, lo, hi).astype(np.float32)
    im = (im - lo) / (hi - lo) * 255.0
    return im.astype(np.uint8)

df = pd.read_csv(f"{ROOT}/data/{split}.csv")
ids = df["id"].unique().tolist()
rows = []
for i in ids:
    m = re.match(r"(case\d+)_(day\d+)_slice_(\d+)$", i)
    case, day, sl = m.group(1), m.group(2), m.group(3)
    rows.append((i, case, day, sl, int(sl), f"{case}_{day}"))
meta = pd.DataFrame(rows, columns=["id", "case", "day", "slice", "slice_int", "case_day"])
meta = meta.sort_values(["case", "day", "slice_int"]).reset_index(drop=True)
meta["gid"] = (meta["case_day"] != meta["case_day"].shift()).cumsum()
meta.to_csv(f"{OUT}/{split}_index.csv", index=False)

N = len(meta)
mm = np.lib.format.open_memmap(f"{OUT}/{split}_img{SIZE}.npy", mode="w+",
                               dtype=np.uint8, shape=(N, SIZE, SIZE))
for k, r in enumerate(meta.itertuples(index=False)):
    p = img_path(split, r.case, r.case_day, r.slice)
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    mm[k] = norm_u8(im)
    if k % 2000 == 0:
        print(k, r.id, flush=True)
mm.flush()
print("done", split, N, "groups", meta.gid.nunique())
