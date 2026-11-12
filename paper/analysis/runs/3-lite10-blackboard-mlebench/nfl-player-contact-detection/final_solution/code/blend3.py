import numpy as np, pandas as pd, os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blend_final import Sub, optimize_thresholds, logit, global_mcc
from blend2 import base_train, base_test, make_smoother

OUT = "workspaces/alkaid_psi"


def load(df, path, col, outcol, test=False):
    if not os.path.exists(path):
        print("missing", path); return df
    d = pd.read_parquet(path)
    d = d[["contact_id", col]].rename(columns={col: outcol})
    d = d.dropna(subset=["contact_id"]).drop_duplicates("contact_id")
    df = df.merge(d, on="contact_id", how="left")
    print("merged", outcol, path, "cov", round(float(df[outcol].notna().mean()), 4))
    return df


RAW_LOGIT = {"capstmix"}


def blend_scores(df, weights):
    n = len(df)
    num = np.zeros(n); den = np.zeros(n)
    for s, w in weights.items():
        z = df[s].values.astype(np.float64) if s in RAW_LOGIT else logit(df[s].values)
        m = np.isfinite(z)
        num[m] += w * z[m]; den[m] += w
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def greedy(df, y, names, keys, wss, rounds=4):
    sm = make_smoother(df, np.ones(len(df), bool), keys)
    cache = {}

    def sc(weights):
        return blend_scores(df, weights)

    best = None
    for s in names:
        if df[s].notna().mean() < 1e-4:
            continue
        for ws in wss:
            score = sm(logit(df[s].values), ws)
            _, m = optimize_thresholds([Sub(y, score, s)])
            if best is None or m > best[0]:
                best = (m, {s: 1.0}, ws, score)
    print("  base", best[0], best[1], "ws", best[2])
    improved = True
    while improved:
        improved = False
        for s in names:
            if s in best[1] or df[s].notna().mean() < 1e-4:
                continue
            for w in [0.25, 0.5, 0.8, 1.2, 1.8]:
                cand = dict(best[1]); cand[s] = w
                z = sc(cand)
                for ws in wss:
                    score = sm(z, ws)
                    _, m = optimize_thresholds([Sub(y, score, s)])
                    if m > best[0] + 1e-5:
                        best = (m, cand, ws, score)
                        improved = True
                        print("  +", s, w, "ws", ws, "->", round(m, 5), best[1])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="final_solution")
    ap.add_argument("--cnn_tag", default="v2m")
    ap.add_argument("--cnn_gr", default="v1m")
    ap.add_argument("--cnn_grB", default="vB1m")
    ap.add_argument("--cnn_grC", default="vC1m")
    args = ap.parse_args()
    tr = base_train(); te = base_test()
    CAN = "workspaces/canopus_pi/preds"
    CAP = "workspaces/capella_pi"
    tr = load(tr, f"{CAN}/ens_oof_probs_v3.parquet", "p_pp", "can_pp")
    tr = load(tr, f"{CAN}/ens_oof_probs_v3.parquet", "p_gr", "can_gr")
    te = load(te, f"{CAN}/ens_test_probs_v3.parquet", "p_pp", "can_pp")
    te = load(te, f"{CAN}/ens_test_probs_v3.parquet", "p_gr", "can_gr")
    tr = load(tr, f"{OUT}/oof_cnn_cand_{args.cnn_tag}.parquet", "pred", "cnn_pp")
    te = load(te, f"{OUT}/test_cnn_cand_{args.cnn_tag}.parquet", "pred", "cnn_pp")
    tr = load(tr, f"{OUT}/oof_ground_cnn_{args.cnn_gr}.parquet", "pred", "cnn_gr")
    te = load(te, f"{OUT}/test_ground_cnn_{args.cnn_gr}.parquet", "pred", "cnn_gr")
    tr = load(tr, f"{OUT}/oof_ground_cnn_{args.cnn_grB}.parquet", "pred", "cnn_grB")
    te = load(te, f"{OUT}/test_ground_cnn_{args.cnn_grB}.parquet", "pred", "cnn_grB")
    tr = load(tr, f"{OUT}/oof_ground_cnn_{args.cnn_grC}.parquet", "pred", "cnn_grC")
    te = load(te, f"{OUT}/test_ground_cnn_{args.cnn_grC}.parquet", "pred", "cnn_grC")
    tr = load(tr, f"{CAP}/oof_lgbm.parquet", "pred", "cap")
    te = load(te, f"{CAP}/test_lgbm.parquet", "pred", "cap")
    tr = load(tr, f"{OUT}/oof_cnn_cand_vW1m.parquet", "pred", "cnn_ppW")
    te = load(te, f"{OUT}/test_cnn_cand_vW1m.parquet", "pred", "cnn_ppW")
    tr = load(tr, f"{CAP}/oof_stack_v7.parquet", "stack_oof", "capst7")
    te = load(te, f"{CAP}/test_stack_v7.parquet", "stack_pred", "capst7")
    tr = load(tr, f"{CAP}/oof_stack_mix.parquet", "stack_oof", "capstmix")
    te = load(te, f"{CAP}/test_stack_mix.parquet", "stack_pred", "capstmix")
    tr = load(tr, f"{CAP}/oof_stack_v6.parquet", "stack_oof", "capst6")
    te = load(te, f"{CAP}/test_stack_v6.parquet", "stack_pred", "capst6")
    tr = load(tr, f"{CAP}/oof_stack_v5.parquet", "stack_oof", "capst5")
    te = load(te, f"{CAP}/test_stack_v5.parquet", "stack_pred", "capst5")
    tr = load(tr, f"{CAP}/oof_stack_v4.parquet", "stack_oof", "capst4")
    te = load(te, f"{CAP}/test_stack_v4.parquet", "stack_pred", "capst4")
    tr = load(tr, f"{CAP}/oof_stack_v3.parquet", "stack_oof", "capst3")
    te = load(te, f"{CAP}/test_stack_v3.parquet", "stack_pred", "capst3")
    tr = load(tr, f"{CAN}/ens_oof_probs_v4.parquet", "p_pp", "can_ens4pp")
    te = load(te, f"{CAN}/ens_test_probs_v4.parquet", "p_pp", "can_ens4pp")
    tr = load(tr, f"{CAN}/ens_oof_probs_v4.parquet", "p_gr", "can_ens4gr")
    te = load(te, f"{CAN}/ens_test_probs_v4.parquet", "p_gr", "can_ens4gr")
    tr = load(tr, f"{CAN}/oof_pp_gru_v3.parquet", "pred", "pp_gru3")
    te = load(te, f"{CAN}/test_pp_gru_v3.parquet", "pred", "pp_gru3")
    tr = load(tr, f"{CAN}/oof_ground_gru_v4.parquet", "pred", "gr_gru4")
    te = load(te, f"{CAN}/test_ground_gru_v4.parquet", "pred", "gr_gru4")
    tr = load(tr, f"{CAN}/oof_ground_gru_v3.parquet", "pred", "gr_gru3")
    te = load(te, f"{CAN}/test_ground_gru_v3.parquet", "pred", "gr_gru3")
    tr = load(tr, f"{CAN}/oof_pp_xgb1.parquet", "pred", "pp_xgb")
    te = load(te, f"{CAN}/test_pp_xgb1.parquet", "pred", "pp_xgb")
    tr = load(tr, f"{CAN}/oof_ground_xgb1.parquet", "pred", "gr_xgb")
    te = load(te, f"{CAN}/test_ground_xgb1.parquet", "pred", "gr_xgb")
    tr = load(tr, f"{CAN}/oof_pp_gru_v2.parquet", "pred", "pp_gru2")
    te = load(te, f"{CAN}/test_pp_gru_v2.parquet", "pred", "pp_gru2")
    tr = load(tr, f"{CAN}/oof_ground_gru_v2.parquet", "pred", "gr_gru2")
    te = load(te, f"{CAN}/test_ground_gru_v2.parquet", "pred", "gr_gru2")
    tr = load(tr, f"{CAN}/oof_pp_v5.parquet", "pred", "can_pp5")
    te = load(te, f"{CAN}/test_pp_v5.parquet", "pred", "can_pp5")
    tr = load(tr, f"{CAN}/oof_ground_v4.parquet", "pred", "can_gr4")
    te = load(te, f"{CAN}/test_ground_v4.parquet", "pred", "can_gr4")
    tr = load(tr, f"{CAN}/oof_pp_gru_v1.parquet", "pred", "pp_gru")
    te = load(te, f"{CAN}/test_pp_gru_v1.parquet", "pred", "pp_gru")
    tr = load(tr, f"{CAN}/oof_ground_gru_v1.parquet", "pred", "gr_gru")
    te = load(te, f"{CAN}/test_ground_gru_v1.parquet", "pred", "gr_gru")
    tr = load(tr, f"{CAP}/oof_stack_v2.parquet", "stack_oof", "capst")
    te = load(te, f"{CAP}/test_stack_v2.parquet", "stack_pred", "capst")
    tr = load(tr, f"{CAP}/oof_stack.parquet", "stack_oof", "capst1")
    te = load(te, f"{CAP}/test_stack.parquet", "stack_pred", "capst1")
    tr.to_parquet(f"{OUT}/blend3_train.parquet", index=False)
    te.to_parquet(f"{OUT}/blend3_test.parquet", index=False)

    pp = ~tr.is_ground.values; gr = tr.is_ground.values
    pp_y = tr.contact.values[pp].astype(np.int8)
    gr_y = tr.contact.values[gr].astype(np.int8)
    pp_keys = ["game_play", "p1", "p2s"]; gr_keys = ["game_play", "p1"]
    pp_names = [c for c in ["capstmix", "capst7", "capst6", "capst5", "capst4", "capst3", "can_ens4pp", "can_pp5", "pp_xgb", "pp_gru3", "pp_gru2", "pp_gru", "cnn_ppW", "capst", "capst1", "can_pp", "cnn_pp", "cap"] if c in tr.columns and tr.loc[pp, c].notna().any()]
    gr_names = [c for c in ["capstmix", "capst7", "capst6", "capst5", "capst4", "capst3", "can_ens4gr", "can_gr4", "gr_gru4", "gr_gru3", "gr_gru2", "gr_gru", "gr_xgb", "capst", "capst1", "can_gr", "cnn_gr", "cnn_grB", "cnn_grC", "cap"] if c in tr.columns and tr.loc[gr, c].notna().any()]
    print("PP sources", pp_names); print("GR sources", gr_names)
    tr_pp = tr[pp].reset_index(drop=True); tr_gr = tr[gr].reset_index(drop=True)
    print("== PP greedy ==")
    bpp = greedy(tr_pp, pp_y, pp_names, pp_keys, [0, 1, 2, 3])
    print("== GR greedy ==")
    bgr = greedy(tr_gr, gr_y, gr_names, gr_keys, [0, 2, 4, 6, 8])
    subs = [Sub(pp_y, bpp[3], "pp"), Sub(gr_y, bgr[3], "gr")]
    ks, M = optimize_thresholds(subs)
    print("FINAL OOF MCC", M, "pp_sub", bpp[0], "gr_sub", bgr[0])
    print("PP", bpp[1], "ws", bpp[2], "th", subs[0].th[ks[0]] if ks[0] >= 0 else "inf")
    print("GR", bgr[1], "ws", bgr[2], "th", subs[1].th[ks[1]] if ks[1] >= 0 else "inf")

    # test
    te_pp = ~te.is_ground.values; te_gr = te.is_ground.values
    te_pp_df = te[te_pp].reset_index(drop=True); te_gr_df = te[te_gr].reset_index(drop=True)
    zpp = blend_scores(te_pp_df, bpp[1])
    zgr = blend_scores(te_gr_df, bgr[1])
    smpp = make_smoother(te, te_pp, pp_keys)
    smgr = make_smoother(te, te_gr, gr_keys)
    spp = smpp(zpp, bpp[2]); sgr = smgr(zgr, bgr[2])
    pred = np.zeros(len(te), np.int8)
    pred[te_pp] = (spp >= subs[0].th[ks[0]]).astype(np.int8) if ks[0] >= 0 else 0
    pred[te_gr] = (sgr >= subs[1].th[ks[1]]).astype(np.int8) if ks[1] >= 0 else 0
    sub = pd.DataFrame({"contact_id": te.contact_id, "contact": pred})
    os.makedirs(args.outdir, exist_ok=True)
    sub.to_csv(f"{args.outdir}/submission.csv", index=False)
    print("wrote", sub.shape, "pos", sub.contact.mean(), "pp_pos", pred[te_pp].mean(), "gr_pos", pred[te_gr].mean())
    # publish blended probs (for stacking / transparency)
    total_pp = pd.Series(np.nan, index=range(len(tr)))
    rep = {"oof_mcc": M, "pp": {"weights": bpp[1], "ws": bpp[2], "sub_mcc": bpp[0],
                                "th": float(subs[0].th[ks[0]])},
           "gr": {"weights": bgr[1], "ws": bgr[2], "sub_mcc": bgr[0],
                  "th": float(subs[1].th[ks[1]])}}
    json.dump(rep, open(f"{OUT}/blend3_report.json", "w"), indent=2)
    print("report saved")
    # save OOF/test blended scores + per-fold robustness
    oof_score = np.full(len(tr), np.nan)
    oof_score[pp] = bpp[3]
    oof_score[gr] = bgr[3]
    oo = pd.DataFrame({"contact_id": tr.contact_id.values, "is_ground": tr.is_ground.values,
                       "fold": tr.fold.values, "contact": tr.contact.values, "score": oof_score})
    oo.to_parquet(f"{OUT}/oof_blend3.parquet", index=False)
    te_score = np.full(len(te), np.nan)
    te_score[te_pp] = spp; te_score[te_gr] = sgr
    pd.DataFrame({"contact_id": te.contact_id.values, "is_ground": te.is_ground.values,
                  "score": te_score}).to_parquet(f"{OUT}/test_blend3_probs.parquet", index=False)
    def mcc_counts(tp, fp, P, N):
        fn = P - tp; tn = N - fp
        d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return 0.0 if d == 0 else (tp * tn - fp * fn) / d
    print("per-fold MCC at global thresholds:")
    for f in range(5):
        m = (tr.fold.values == f)
        mtp = mfp = 0; MP = MN = 0
        for sb, k, sub_mask in [(subs[0], ks[0], pp), (subs[1], ks[1], gr)]:
            sm = m & sub_mask
            if sm.sum() == 0:
                continue
            y = tr.contact.values[sm]; sc = oof_score[sm]
            tp = int(((sc >= (sb.th[k] if k >= 0 else np.inf)) & (y == 1)).sum())
            fp = int(((sc >= (sb.th[k] if k >= 0 else np.inf)) & (y == 0)).sum())
            mtp += tp; mfp += fp; MP += int(y.sum()); MN += int((1 - y).sum())
        print(f"  fold{f} MCC {mcc_counts(mtp, mfp, MP, MN):.4f} (P={MP}, N={MN})")
    import pickle
    json.dump({"pp": {"weights": bpp[1], "ws": bpp[2], "th": float(subs[0].th[ks[0]])},
               "gr": {"weights": bgr[1], "ws": bgr[2], "th": float(subs[1].th[ks[1]])}},
              open(f"{OUT}/blend3_apply.json", "w"), indent=2)


if __name__ == "__main__":
    main()
