import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ground_lib import params_single

VIEWS = ["Sideline", "Endzone"]
OUT = "workspaces/alkaid_psi"
IMH = 720.0


def build(plan_path, out_path, variant="A"):
    df = pd.read_parquet(plan_path).reset_index(drop=True)
    n = len(df)
    parts = []
    for v in VIEWS:
        b = df[[f"{v}_b_{k}" for k in range(4)]].values.astype(np.float32)
        nv = df[f"{v}_nv"].values.astype(np.float32)
        g = np.zeros((n, 9), np.float32)
        for i in range(n):
            bi = b[i] if nv[i] > 0 else None
            cx, cy, side, k = params_single(bi, variant)
            g[i, 0] = k
            if side > 0 and bi is not None:
                g[i, 1] = (bi[0] - cx) / side + 0.5
                g[i, 2] = (bi[1] - cy) / side + 0.5
                g[i, 3] = bi[2] / side
                g[i, 4] = bi[3] / side
                g[i, 5] = bi[1] / IMH
                g[i, 6] = bi[3] / IMH
                g[i, 7] = side / IMH
                g[i, 8] = bi[0] / 1280.0
        parts.append(g)
    geom = np.concatenate(parts, axis=1).astype(np.float32)
    np.save(out_path, geom)
    print(plan_path, "->", out_path, geom.shape)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="A"); a = ap.parse_args()
    pref = "ground" if a.variant == "A" else f"ground{a.variant}"
    build(f"{OUT}/train_ground.parquet", f"{OUT}/geom_{pref}_train.npy", a.variant)
    build(f"{OUT}/test_ground.parquet", f"{OUT}/geom_{pref}_test.npy", a.variant)
