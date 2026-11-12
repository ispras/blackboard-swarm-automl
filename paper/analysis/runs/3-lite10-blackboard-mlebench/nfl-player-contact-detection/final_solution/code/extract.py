import numpy as np, cv2, pandas as pd, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_lib import params_from_boxes, crop

S = 112
VIEWS = ["Sideline", "Endzone"]
OUT = "workspaces/alkaid_psi"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
NW = int(os.environ.get("NW", "4"))
PPVAR = os.environ.get("PPVAR", "T")
BASE = f"{OUT}/{SPLIT}_crops" if PPVAR == "T" else f"{OUT}/{SPLIT}_pp{PPVAR}_crops"
PLAN = f"{OUT}/{SPLIT}_cand.parquet"
ARR = BASE + ".npy"
VDIR = "data/train" if SPLIT == "train" else "data/test"


def boxes_of(r, v):
    b1 = r[f"{v}_b1_0"], r[f"{v}_b1_1"], r[f"{v}_b1_2"], r[f"{v}_b1_3"]
    b2 = r[f"{v}_b2_0"], r[f"{v}_b2_1"], r[f"{v}_b2_2"], r[f"{v}_b2_3"]
    nv = int(r[f"{v}_nv"])
    if nv == 0:
        return None, None
    if nv == 1:
        if b1[2] > 0:
            return np.array(b1, np.float32), None
        return None, np.array(b2, np.float32)
    return np.array(b1, np.float32), np.array(b2, np.float32)


def process_play(args):
    gp, plan, arrpath, N = args
    arr = np.lib.format.open_memmap(arrpath, mode="r+")
    d = plan[plan.game_play == gp]
    for vi, view in enumerate(VIEWS):
        dv = d[d[f"{view}_nv"] > 0]
        if len(dv) == 0:
            continue
        # rows grouped by frame
        groups = dv.groupby("frame")
        need = set(groups.groups.keys())
        cap = cv2.VideoCapture(os.path.join(VDIR, f"{gp}_{view}.mp4"))
        if not cap.isOpened():
            print("CANNOT OPEN", gp, view, flush=True)
            continue
        maxf = max(need)
        fi = 0
        while fi <= maxf:
            ok = cap.grab()
            if not ok:
                break
            if fi in need:
                ok2, frame = cap.retrieve()
                if ok2:
                    sub = groups.get_group(fi)
                    for _, r in sub.iterrows():
                        b1, b2 = boxes_of(r, view)
                        cx, cy, side, nv = params_from_boxes(b1, b2, PPVAR)
                        arr[int(r["row"]), vi] = crop(frame, cx, cy, side, S)
            fi += 1
        cap.release()
    arr.flush()
    return gp


if __name__ == "__main__":
    t = time.time()
    plan = pd.read_parquet(PLAN)
    plan["row"] = np.arange(len(plan), dtype=np.int32)
    N = len(plan)
    if not os.path.exists(ARR):
        arr = np.lib.format.open_memmap(ARR, mode="w+", dtype=np.uint8, shape=(N, len(VIEWS), S, S, 3))
        for i in range(0, N, 20000):
            arr[i:i+20000] = 0
        del arr
    plays = sorted(plan.game_play.unique())
    print(f"{SPLIT}: {N} rows, {len(plays)} plays, workers={NW}", flush=True)
    with Pool(NW) as p:
        for k, gp in enumerate(p.imap_unordered(process_play, [(g, plan, ARR, N) for g in plays])):
            if k % 20 == 0:
                print(f"done {k}/{len(plays)} {gp} {time.time()-t:.0f}s", flush=True)
    plan[["contact_id","game_play","frame","row"]].to_parquet(f"{OUT}/{SPLIT}_crop_index.parquet", index=False)
    print("DONE", time.time() - t, flush=True)
