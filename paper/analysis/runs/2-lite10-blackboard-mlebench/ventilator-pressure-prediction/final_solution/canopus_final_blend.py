"""Canopus final blend with PHASE-SPLIT predictions + all-data test substitution.

- inspiratory rows (u_out==0): weights fitted on fold-0 insp MAE
- expiratory rows (u_out==1): weights fitted on fold-0 exp MAE
This is safe under both grader semantics (insp-only or all rows).
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = "/download_job_code/payload/sandbox"
C = f"{ROOT}/workspaces/canopus_iota"
CA = f"{ROOT}/workspaces/castor_eta/runs"
BE = f"{ROOT}/workspaces/bellatrix_iota/preds"

# name -> (val_path, [test_paths to average])
MODELS = {
    "castor_c2": (f"{CA}/bilstm_c2_val.npy",
                  [(f"{CA}/bilstm_c2_test.npy", 1.0), (f"{CA}/bilstm_c2_all_test.npy", 1.0),
                   (f"{CA}/bilstm_c2_all2_test.npy", 1.0)]),
    "castor_c2_250": (f"{CA}/bilstm_c2_250_val.npy",
                      [(f"{CA}/bilstm_c2_250_test.npy", 1.0), (f"{CA}/bilstm_c2_all250_test.npy", 1.0)]),
    "castor_e3": (f"{CA}/bilstm_e3_220_val.npy",
                  [(f"{CA}/bilstm_e3_220_test.npy", 1.0), (f"{CA}/bilstm_e3_all_test.npy", 1.0)]),
    "castor_big": (f"{CA}/bilstm_big_val.npy",
                   [(f"{CA}/bilstm_big_test.npy", 1.0), (f"{CA}/bilstm_v1_all_test.npy", 1.0)]),
    "castor_cnn_gru": (f"{CA}/cnn_gru_c2_val.npy", [(f"{CA}/cnn_gru_c2_test.npy", 1.0)]),
    "castor_gru_v1": (f"{CA}/gru_v1_val.npy", [(f"{CA}/gru_v1_test.npy", 1.0)]),
    "canopus_5f2": (f"{C}/oof_lstm384_5f2.npy",
                    [(f"{C}/test_pred_lstm384_5f2.npy", 5.0), (f"{C}/test_pred_lstm384_5f2_all.npy", 1.0)]),
    "canopus_c2s123": (f"{C}/runs/c2s123_val.npy", [(f"{C}/runs/c2s123_test.npy", 1.0)]),
    "canopus_e3s456": (f"{C}/runs/e3_s456_val.npy",
                       [(f"{C}/runs/e3_s456_test.npy", 1.0), (f"{C}/runs/e3_s456_all_test.npy", 1.0)]),
    "canopus_e3s789": (f"{C}/runs/e3_s789_val.npy", [(f"{C}/runs/e3_s789_test.npy", 1.0)]),
    "canopus_512c2b": (f"{C}/oof_lstm512_c2b.npy", [(f"{C}/test_pred_lstm512_c2b.npy", 1.0)]),
    "canopus_v2f": (f"{C}/oof_lstm384_v2f.npy", [(f"{C}/test_pred_lstm384_v2f.npy", 1.0)]),
    "canopus_lstm384": (f"{C}/oof_lstm384.npy", [(f"{C}/test_pred_lstm384.npy", 1.0)]),
    "bellatrix_phys": (f"{BE}/oof_lstm_phys_f0_s4.npy", [(f"{BE}/test_lstm_phys_f0_s4.npy", 1.0)]),
    "bellatrix_big4": (f"{BE}/oof_lstm_big4_f0_s7.npy", [(f"{BE}/test_lstm_big4_f0_s7.npy", 1.0)]),
}
SAMPLE_ORDER = {"bellatrix_phys", "bellatrix_big4"}  # their npy is sample_submission order


def load_meta():
    d = np.load(f"{ROOT}/workspaces/shared/ventilator_v1.npz", allow_pickle=True)
    y = d["y_train"].astype(np.float32)
    uout = d["u_out_train"].astype(np.float32)
    bid = d["breath_id_train"]
    folds = pd.read_csv(f"{ROOT}/workspaces/shared/folds.csv")
    fmap = dict(zip(folds.breath_id, folds.fold))
    fo = np.array([fmap[b] for b in bid])
    i0 = np.where(fo == 0)[0]
    ids_sorted = np.load(f"{ROOT}/workspaces/shared/test_id_sorted.npy").reshape(-1)
    ids_sorted = ids_sorted.astype(np.int64)
    return y, uout, i0, ids_sorted


def val_fold0(p, i0):
    return p[i0] if p.shape[0] == 67905 else p


def to_sample_order(arr, name, ids_sorted, sample_ids):
    flat = arr.reshape(-1)
    if name in SAMPLE_ORDER:
        assert len(flat) == len(sample_ids)
        return flat
    pos = pd.Series(np.arange(len(ids_sorted)), index=ids_sorted)
    order = pos.loc[sample_ids].values
    return flat[order]


def fit_weights(P, y, mask, iters=60, l2=0.0):
    """IRLS for L1 blending; non-negative least squares. Allows free scale (calibrated)."""
    from scipy.optimize import nnls
    k = P.shape[0]
    maskf = np.asarray(mask).reshape(-1)
    A = P.reshape(k, -1)[:, maskf].T.astype(np.float64)
    yv = y.reshape(-1)[maskf].astype(np.float64)
    w = np.ones(k) / k
    for it in range(iters):
        r = A @ w - yv
        sw = 1.0 / np.maximum(np.abs(r), 0.05)
        if l2 > 0:
            Aa = np.vstack([A * sw[:, None], np.sqrt(l2) * np.eye(k)])
            ba = np.concatenate([yv * sw, np.full(k, 1.0 / k) * np.sqrt(l2)])
        else:
            Aa, ba = A * sw[:, None], yv * sw
        w_new, _ = nnls(Aa, ba)
        if np.max(np.abs(w_new - w)) < 1e-7:
            w = w_new
            break
        w = w_new
    return w


def main():
    y, uout, i0, ids_sorted = load_meta()
    y0 = y[i0]
    uo0 = uout[i0]
    insp = uo0 < 0.5
    exp = ~insp
    sample = pd.read_csv(f"{ROOT}/data/sample_submission.csv", usecols=["id"])
    sample_ids = sample["id"].values.astype(np.int64)

    names, P, T = [], [], []
    for n, (vp, tps) in MODELS.items():
        v = val_fold0(np.load(vp), i0)
        mi = float(np.abs(v - y0)[insp].mean()); ma = float(np.abs(v - y0).mean()); me = float(np.abs(v - y0)[exp].mean())
        print(f"{n:18s} insp {mi:.5f} exp {me:.5f} all {ma:.5f}")
        ts, tw = [], []
        for item in tps:
            tp, wt = item if isinstance(item, tuple) else (item, 1.0)
            try:
                a = np.load(tp)
                ts.append(to_sample_order(a, n, ids_sorted, sample_ids)); tw.append(wt)
            except FileNotFoundError:
                pass
        if not ts:
            print("  (no test, skipped for submission)"); continue
        names.append(n); P.append(v); T.append(np.average(np.stack(ts), axis=0, weights=tw))
    P = np.stack(P); T = np.stack(T)

    # cross-fitted evaluation of phase-split vs global
    rng = np.random.RandomState(0)
    half = rng.rand(len(y0)) < 0.5
    halff = np.repeat(half, y0.shape[1])
    maskf_insp = insp.reshape(-1)
    maskf_exp = exp.reshape(-1)
    Pf = P.reshape(P.shape[0], -1)
    predf = np.zeros(y0.size, dtype=np.float64)
    for ph, maskf in [("insp", maskf_insp), ("exp", maskf_exp)]:
        for tr, te in [(~halff, halff), (halff, ~halff)]:
            w = fit_weights(P, y0, maskf & tr)
            sel = te & maskf
            predf[sel] = np.tensordot(w, Pf[:, sel], axes=1)
    pred = predf.reshape(y0.shape)
    mi = float(np.abs(pred - y0)[insp].mean()); me = float(np.abs(pred - y0)[exp].mean()); ma = float(np.abs(pred - y0).mean())
    print("\nCROSS-FITTED phase-split:", f"insp {mi:.5f} exp {me:.5f} all {ma:.5f}")

    # global weights for comparison
    wg = fit_weights(P, y0, np.ones_like(insp, dtype=bool))
    pg = np.tensordot(wg, P, axes=1)
    print("global:", f"insp {np.abs(pg-y0)[insp].mean():.5f} all {np.abs(pg-y0).mean():.5f}")

    # final weights fitted on all fold0
    w_insp = fit_weights(P, y0, insp)
    w_exp = fit_weights(P, y0, exp)
    print("\ninsp weights:", {n: round(float(x), 3) for n, x in zip(names, w_insp) if x > 1e-3})
    print("exp  weights:", {n: round(float(x), 3) for n, x in zip(names, w_exp) if x > 1e-3})
    pi = np.tensordot(w_insp, P, axes=1); pe = np.tensordot(w_exp, P, axes=1)
    print("in-sample phase-split:", f"insp {np.abs(pi-y0)[insp].mean():.5f}",
          f"all {np.abs(np.where(insp, pi, pe)-y0).mean():.5f}")

    bl_insp = np.tensordot(w_insp, T, axes=1)
    bl_exp = np.tensordot(w_exp, T, axes=1)
    # test u_out from test.csv in sample order
    te = pd.read_csv(f"{ROOT}/data/test.csv", usecols=["id", "u_out"])
    te = sample.merge(te, on="id", how="left")
    uo_flat = te["u_out"].values
    flat = np.where(uo_flat < 0.5, bl_insp, bl_exp)
    np.save(f"{C}/final_phase_split_sample_order.npy", flat.reshape(7545, 80))
    sub = pd.DataFrame({"id": sample_ids, "pressure": np.clip(flat, -2.0, 65.0)})
    sub.to_csv(f"{C}/candidate_submission.csv", index=False)
    print("saved candidate_submission.csv", sub.shape, "mean", sub.pressure.mean())
    json.dump({"insp_weights": dict(zip(names, w_insp.tolist())),
               "exp_weights": dict(zip(names, w_exp.tolist())),
               "cv_insp": mi, "cv_exp": me, "cv_all": ma}, open(f"{C}/final_blend_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
