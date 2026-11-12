import argparse, time, json
import numpy as np
from train import run_fold, load_jsonl

SCORED_IDX = [0, 1, 3]


def build_oof(outdir, folds_file, n=2160):
    folds = np.load(folds_file)
    nfold = int(folds.max()) + 1
    oof = np.zeros((n, 107, 5), dtype=np.float32)
    tests = []
    for f in range(nfold):
        vi = np.load(f"{outdir}/va_idx_fold{f}.npy")
        oof[vi] = np.load(f"{outdir}/oof_fold{f}.npy")
        tests.append(np.load(f"{outdir}/test_fold{f}.npy"))
    test = np.mean(tests, axis=0).astype(np.float32)
    np.save(f"{outdir}/oof.npy", oof)
    np.save(f"{outdir}/test.npy", test)
    return oof, test


def eval_oof(oof):
    d = load_jsonl("data/train.json")
    sn = np.array([x.get("SN_filter", 1) == 1 for x in d])
    true = np.zeros_like(oof)
    for i, x in enumerate(d):
        for j, t in enumerate(["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]):
            true[i, :68, SCORED_IDX[j]] = x[t]
    def m(sel):
        se = ((oof[sel][:, :68][:, :, SCORED_IDX] - true[sel][:, :68][:, :, SCORED_IDX]) ** 2).mean((0, 1))
        return np.sqrt(se).mean()
    return m(np.ones(len(d), bool)), m(sn), sn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--scored_only", action="store_true")
    ap.add_argument("--clip_aux", type=float, default=0.0)
    ap.add_argument("--sn_only", action="store_true")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--nblocks", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--select", default="sn1", choices=["sn1", "all"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--nfold", type=int, default=5)
    ap.add_argument("--asinh_s", type=float, default=0.0)
    a = ap.parse_args()
    folds_file = ("blackboard/mirfak_zeta_cv_split.npy" if a.nfold == 5
                  else "blackboard/mirfak_zeta_cv_split10.npy")
    t0 = time.time()
    for seed in a.seeds:
        for f in range(a.nfold):
            run_fold(f, a.epochs, seed, a.outdir, a.lr, a.batch,
                     use_all_targets=not a.scored_only, clip_aux=a.clip_aux,
                     sn_only=a.sn_only, aug=not a.no_aug, hidden=a.hidden,
                     nblocks=a.nblocks, drop=a.drop, kernel=a.kernel,
                     patience=a.patience, select=a.select, tta=a.tta, mixup=a.mixup,
                     folds_file=folds_file, asinh_s=a.asinh_s)
    oof, test = build_oof(a.outdir, folds_file)
    allm, snm, _ = eval_oof(oof)
    allm, snm = float(allm), float(snm)
    print(f"CV DONE {a.outdir} all={allm:.4f} SN1={snm:.4f} time={time.time()-t0:.1f}s", flush=True)
    with open(f"{a.outdir}/summary.json", "w") as fp:
        json.dump({"all": allm, "sn1": snm}, fp)
