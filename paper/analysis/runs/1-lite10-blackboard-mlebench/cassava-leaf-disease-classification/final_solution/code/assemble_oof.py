import argparse, json
import numpy as np
import pandas as pd

ROOT = "/download_job_code/payload/sandbox"
SPLIT = f"{ROOT}/workspaces/shared/folds5_seed42.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--split", default=SPLIT)
    args = ap.parse_args()

    df = pd.read_csv(args.split)
    n = len(df)
    oof = np.full((n, 5), np.nan)
    folds = [int(x) for x in args.folds.split(",")]
    for f in folds:
        vp = np.load(f"{args.rundir}/f{f}_val_probs.npy")
        idx = np.where(df.fold.values == f)[0]
        assert len(idx) == len(vp), (f, len(idx), len(vp))
        oof[idx] = vp
    valid = ~np.isnan(oof).any(1)
    acc = (oof[valid].argmax(1) == df.label.values[valid]).mean()
    np.save(f"{args.rundir}/oof_{args.tag}.npy", oof)
    tp = np.mean([np.load(f"{args.rundir}/f{f}_test_probs.npy") for f in folds], 0)
    np.save(f"{args.rundir}/test_{args.tag}.npy", tp)
    # fold-0 val probs for shared-holdout ensembling (order of df[df.fold==0])
    f0 = np.where(df.fold.values == 0)[0]
    np.save(f"{args.rundir}/val0_{args.tag}.npy", oof[f0])
    y0 = df.label.values[f0]
    print(json.dumps({"tag": args.tag, "oof_acc_all": float(acc),
                      "val0_acc": float((oof[f0].argmax(1) == y0).mean()),
                      "rows_filled": int(valid.sum())}))


if __name__ == "__main__":
    main()
