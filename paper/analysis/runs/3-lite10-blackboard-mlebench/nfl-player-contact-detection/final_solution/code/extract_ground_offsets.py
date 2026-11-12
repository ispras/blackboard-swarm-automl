import numpy as np, cv2, pandas as pd, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_lib import crop
from ground_lib import params_single

S = 112
VIEWS = ["Sideline", "Endzone"]
OUT = "workspaces/alkaid_psi"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
OFF = int(sys.argv[2]) if len(sys.argv) > 2 else -6
VARIANT = os.environ.get("GVAR", "A")
PREFIX = "ground" if VARIANT == "A" else f"ground{VARIANT}"
NW = int(os.environ.get("NW", "4"))
PLAN = f"{OUT}/{SPLIT}_ground.parquet"
ARR = f"{OUT}/{SPLIT}_{PREFIX}_crops_off{OFF}.npy"
VDIR = "data/train" if SPLIT == "train" else "data/test"


def process_play(args):
    gp, plan, arrpath = args
    arr = np.lib.format.open_memmap(arrpath, mode="r+")
    d = plan[plan.game_play == gp]
    for vi, view in enumerate(VIEWS):
        dv = d[d[f"{view}_nv"] > 0].copy()
        if len(dv) == 0:
            continue
        dv["tframe"] = dv.frame + OFF
        groups = dv.groupby("tframe")
        need = set(groups.groups.keys())
        cap = cv2.VideoCapture(os.path.join(VDIR, f"{gp}_{view}.mp4"))
        if not cap.isOpened():
            print("CANNOT OPEN", gp, view, flush=True)
            continue
        maxf = max(need); fi = 0
        while fi <= maxf:
            ok = cap.grab()
            if not ok:
                break
            if fi in need:
                ok2, frame = cap.retrieve()
                if ok2:
                    for _, r in groups.get_group(fi).iterrows():
                        b = np.array([r[f"{view}_b_{k}"] for k in range(4)], np.float32) if r[f"{view}_nv"] > 0 else None
                        cx, cy, side, nv = params_single(b, VARIANT)
                        if nv:
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
    print(f"{SPLIT} {PREFIX} OFF={OFF}: {N} rows", flush=True)
    with Pool(NW) as p:
        for k, gp in enumerate(p.imap_unordered(process_play, [(g, plan, ARR) for g in plays])):
            if k % 40 == 0:
                print(f"done {k}/{len(plays)} {time.time()-t:.0f}s", flush=True)
    print("DONE", time.time() - t, flush=True)
