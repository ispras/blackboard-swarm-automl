import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_lib import params_from_boxes

VIEWS = ["Sideline", "Endzone"]
OUT = "workspaces/alkaid_psi"


def build(plan_path, out_path, variant="T"):
    df = pd.read_parquet(plan_path).reset_index(drop=True)
    n = len(df)
    G = []
    dist = df.dist.values.astype(np.float32)
    for v in VIEWS:
        b1 = df[[f"{v}_b1_{k}" for k in range(4)]].values.astype(np.float32)
        b2 = df[[f"{v}_b2_{k}" for k in range(4)]].values.astype(np.float32)
        nv = df[f"{v}_nv"].values.astype(np.float32)
        g = np.zeros((n, 11), np.float32)
        for i in range(n):
            nvi = int(nv[i])
            c1 = b1[i] if nvi >= 1 and b1[i, 2] > 0 else None
            c2 = b2[i] if nvi >= 2 else None
            if nvi == 1 and c2 is None and b1[i, 2] == 0:
                c2 = b2[i]
            cx, cy, side, k = params_from_boxes(c1, c2, variant)
            g[i, 0] = k / 2.0
            if side <= 0:
                continue
            if c1 is not None:
                g[i, 1] = 1
                g[i, 2] = (c1[0] - cx) / side + 0.5
                g[i, 3] = (c1[1] - cy) / side + 0.5
                g[i, 4] = c1[2] / side
                g[i, 5] = c1[3] / side
            if c2 is not None:
                g[i, 6] = 1
                g[i, 7] = (c2[0] - cx) / side + 0.5
                g[i, 8] = (c2[1] - cy) / side + 0.5
                g[i, 9] = c2[2] / side
                if c1 is not None:
                    g[i, 10] = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5 / side
        G.append(g)
    geom = np.concatenate([dist[:, None]] + G, axis=1).astype(np.float32)
    names = ["dist_yards"] + [f"{v}_{k}" for v in VIEWS for k in
                              ["nv","has1","u1","v1","w1","h1","has2","u2","v2","w2","cdist"]]
    np.save(out_path, geom)
    print(plan_path, "->", out_path, geom.shape, "nan", np.isnan(geom).any())
    return names


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="T"); a = ap.parse_args()
    sfx = "" if a.variant == "T" else f"_pp{a.variant}"
    build(f"{OUT}/train_cand.parquet", f"{OUT}/geom{sfx}_train.npy", a.variant)
    build(f"{OUT}/test_cand.parquet", f"{OUT}/geom{sfx}_test.npy", a.variant)
