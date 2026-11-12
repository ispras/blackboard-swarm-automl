import os, time
import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"
V2 = os.path.join(ROOT, "workspaces/mimosa_omicron", "feat_v2")
OUT = os.path.join(ROOT, "workspaces/mimosa_omicron")
K = 10.0
NFOLD = 5


def main():
    t0 = time.time()
    pl = sorted(os.path.join(V2, f) for f in os.listdir(V2) if f.endswith(".parquet"))
    ys, ois, pcs, dcs, rts = [], [], [], [], []
    for p in pl:
        df = pd.read_parquet(p, columns=["orig_idx", "fare_amount", "p_cell", "d_cell", "route"])
        ys.append(df["fare_amount"].to_numpy(np.float32))
        ois.append(df["orig_idx"].to_numpy(np.int64))
        pcs.append(df["p_cell"].to_numpy(np.int64))
        dcs.append(df["d_cell"].to_numpy(np.int64))
        rts.append(df["route"].to_numpy(np.int64))
    y = np.concatenate(ys); oi = np.concatenate(ois)
    pc = np.concatenate(pcs); dc = np.concatenate(dcs); rt = np.concatenate(rts)
    del ys, ois, pcs, dcs, rts
    n = len(y)
    fold = (oi % NFOLD).astype(np.int8)
    gmean = float(y.mean())
    print("loaded", n, "global_mean", gmean, time.time() - t0, flush=True)

    res = {"gmean": np.float64(gmean), "k": np.float64(K)}
    for name, arr in [("pcell", pc), ("dcell", dc), ("route", rt)]:
        u, codes = np.unique(arr, return_inverse=True)
        del arr
        ne = len(u)
        sum_all = np.bincount(codes, weights=y, minlength=ne)
        cnt_all = np.bincount(codes, minlength=ne)
        enc_full = ((sum_all + K * gmean) / (cnt_all + K)).astype(np.float32)
        oof = np.empty(n, dtype=np.float32)
        for f in range(NFOLD):
            m = fold == f
            ym = y[m]
            cm = codes[m]
            sum_f = sum_all - np.bincount(cm, weights=ym, minlength=ne)
            cnt_f = cnt_all - np.bincount(cm, minlength=ne)
            prior_f = float(y[~m].mean())
            enc_f = ((sum_f + K * prior_f) / (cnt_f + K)).astype(np.float32)
            oof[m] = enc_f[cm]
        res[f"te_{name}_oof"] = oof
        res[f"u_{name}"] = u
        res[f"enc_{name}_full"] = enc_full
        res[f"cnt_{name}"] = cnt_all.astype(np.float32)
        print(f"{name}: unique={ne} oof_mean={oof.mean():.3f} done {time.time()-t0:.0f}s", flush=True)
        del codes, sum_all, cnt_all, enc_full, oof
    np.savez_compressed(os.path.join(OUT, "te_maps.npz"), **res)
    print("saved", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
