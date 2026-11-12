"""Build the swarm final submission: equal-weight geometric mean of member test probs.

Members are given as  name=TEST_PROBS.npy[:VAL_PROBS.npy]
VAL probs (if given) are fold-0 val probs (df[fold==0] order) used only for reporting.
"""
import os, json, argparse
import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"
SPLIT = f"{ROOT}/workspaces/shared/folds5_seed42.csv"
SAMPLE = f"{ROOT}/data/sample_submission.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="name=test.npy[:val.npy]")
    ap.add_argument("--out", default=f"{ROOT}/final_solution/submission.csv")
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--mode", choices=["arith", "geom"], default="arith")
    args = ap.parse_args()

    def sanitize(P):
        # replace NaN entries (per image/class) with the mean over finite members,
        # then anything still NaN (whole column) with 1/5. Prevents NaN poisoning the union.
        with np.errstate(invalid="ignore"):
            fill = np.nanmean(P, axis=0, keepdims=True)
        P = np.where(np.isnan(P), fill, P)
        P = np.where(np.isnan(P), 0.2, P)
        return P

    def combine(P, eps):
        P = sanitize(P)
        if args.mode == "geom":
            return np.exp(np.log(P + eps).mean(0))
        return P.mean(0)

    df = pd.read_csv(SPLIT)
    f0 = np.where(df.fold.values == 0)[0]
    y0 = df.label.values[f0]
    sub = pd.read_csv(SAMPLE)
    N = len(sub)

    names, tests, vals = [], [], []
    for spec in args.members:
        left = spec.split("=", 1)
        name = left[0]
        parts = left[1].split(":")
        tp = np.load(parts[0])
        if tp.shape[0] != N:
            print(f"SKIP {name}: test shape {tp.shape} != {N}")
            continue
        names.append(name); tests.append(tp.astype(np.float64))
        vp = np.load(parts[1]) if len(parts) > 1 and parts[1] and os.path.exists(parts[1]) else None
        vals.append(vp)

    T = np.stack(tests, 0)  # (M,N,5)
    prob = combine(T, args.eps)
    pred = prob.argmax(1)
    out = pd.DataFrame({"image_id": sub.image_id.values, "label": pred.astype(int)})
    out.to_csv(args.out, index=False)

    report = {"members": names, "n_members": len(names)}
    # fold-0 check where val provided and length matches
    V = []
    for nm, vp in zip(names, vals):
        if vp is not None and vp.shape[0] == len(f0):
            V.append((nm, vp.astype(np.float64)))
    if V:
        PV = np.stack([v for _, v in V], 0)
        gv = combine(PV, args.eps)
        acc = float((gv.argmax(1) == y0).mean())
        report[f"fold0_{args.mode}_acc_subset"] = acc
        report["fold0_members"] = [n for n, _ in V]
        for n, v in V:
            report[f"single_{n}"] = float((v.argmax(1) == y0).mean())
    print(json.dumps(report, indent=2))
    json.dump(report, open(args.out.replace(".csv", "_meta.json"), "w"), indent=2)
    print("wrote", args.out, out.shape, "label dist", np.bincount(pred, minlength=5).tolist())


if __name__ == "__main__":
    main()
