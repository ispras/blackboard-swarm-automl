import numpy as np, pandas as pd, os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blend_final import Sub, optimize_thresholds, logit, global_mcc
from blend2 import base_train, base_test, make_smoother

OUT = "workspaces/alkaid_psi"
CAN = "workspaces/canopus_pi/preds"
CAP = "workspaces/capella_pi"
RAW_LOGIT = {"capstmix", "capstmix2", "capstmix3"}

REG = {
    "capstmix3": (f"{CAP}/oof_stack_mixv3.parquet", "stack_oof", f"{CAP}/test_stack_mixv3.parquet", "stack_pred", True),
    "capstmix2": (f"{CAP}/oof_stack_mixv2.parquet", "stack_oof", f"{CAP}/test_stack_mixv2.parquet", "stack_pred", True),
    "capstmix": (f"{CAP}/oof_stack_mix.parquet", "stack_oof", f"{CAP}/test_stack_mix.parquet", "stack_pred", True),
    "capst8": (f"{CAP}/oof_stack_v8.parquet", "stack_oof", f"{CAP}/test_stack_v8.parquet", "stack_pred", False),
    "capst7": (f"{CAP}/oof_stack_v7.parquet", "stack_oof", f"{CAP}/test_stack_v7.parquet", "stack_pred", False),
    "capst5": (f"{CAP}/oof_stack_v5.parquet", "stack_oof", f"{CAP}/test_stack_v5.parquet", "stack_pred", False),
    "capst4": (f"{CAP}/oof_stack_v4.parquet", "stack_oof", f"{CAP}/test_stack_v4.parquet", "stack_pred", False),
    "can_ens4pp": (f"{CAN}/ens_oof_probs_v4.parquet", "p_pp", f"{CAN}/ens_test_probs_v4.parquet", "p_pp", False),
    "can_ens4gr": (f"{CAN}/ens_oof_probs_v4.parquet", "p_gr", f"{CAN}/ens_test_probs_v4.parquet", "p_gr", False),
    "pp_gru4": (f"{CAN}/oof_pp_gru_v4.parquet", "pred", f"{CAN}/test_pp_gru_v4.parquet", "pred", False),
    "pp_gru3": (f"{CAN}/oof_pp_gru_v3.parquet", "pred", f"{CAN}/test_pp_gru_v3.parquet", "pred", False),
    "pp_gru": (f"{CAN}/oof_pp_gru_v1.parquet", "pred", f"{CAN}/test_pp_gru_v1.parquet", "pred", False),
    "can_pp5": (f"{CAN}/oof_pp_v5.parquet", "pred", f"{CAN}/test_pp_v5.parquet", "pred", False),
    "pp_xgb": (f"{CAN}/oof_pp_xgb1.parquet", "pred", f"{CAN}/test_pp_xgb1.parquet", "pred", False),
    "cnn_pp": (f"{OUT}/oof_cnn_cand_v2m.parquet", "pred", f"{OUT}/test_cnn_cand_v2m.parquet", "pred", False),
    "cnn_ppR": (f"{OUT}/oof_cnn_cand_vR1m.parquet", "pred", f"{OUT}/test_cnn_cand_vR1m.parquet", "pred", False),
    "cap": (f"{CAP}/oof_lgbm.parquet", "pred", f"{CAP}/test_lgbm.parquet", "pred", False),
    "gr_gru5": (f"{CAN}/oof_ground_gru_v5.parquet", "pred", f"{CAN}/test_ground_gru_v5.parquet", "pred", False),
    "gr_gru4": (f"{CAN}/oof_ground_gru_v4.parquet", "pred", f"{CAN}/test_ground_gru_v4.parquet", "pred", False),
    "gr_gru3": (f"{CAN}/oof_ground_gru_v3.parquet", "pred", f"{CAN}/test_ground_gru_v3.parquet", "pred", False),
    "gr_gru2": (f"{CAN}/oof_ground_gru_v2.parquet", "pred", f"{CAN}/test_ground_gru_v2.parquet", "pred", False),
    "can_gr4": (f"{CAN}/oof_ground_v4.parquet", "pred", f"{CAN}/test_ground_v4.parquet", "pred", False),
    "gr_xgb": (f"{CAN}/oof_ground_xgb1.parquet", "pred", f"{CAN}/test_ground_xgb1.parquet", "pred", False),
    "cnn_gr": (f"{OUT}/oof_ground_cnn_v1m.parquet", "pred", f"{OUT}/test_ground_cnn_v1m.parquet", "pred", False),
    "cnn_grB": (f"{OUT}/oof_ground_cnn_vB1m.parquet", "pred", f"{OUT}/test_ground_cnn_vB1m.parquet", "pred", False),
    "cnn_grC": (f"{OUT}/oof_ground_cnn_vC1m.parquet", "pred", f"{OUT}/test_ground_cnn_vC1m.parquet", "pred", False),
}
PP_SRC = ["capstmix3", "capstmix2", "capstmix", "capst8", "capst7", "capst5", "capst4", "pp_gru4", "can_ens4pp", "pp_gru3", "pp_gru", "can_pp5", "pp_xgb", "cnn_pp", "cnn_ppR", "cap"]
GR_SRC = ["capstmix", "capst8", "capst7", "capst5", "capst4", "can_ens4gr", "gr_gru5", "gr_gru4", "gr_gru3", "gr_gru2", "can_gr4", "gr_xgb", "cnn_gr", "cnn_grB", "cnn_grC", "cap"]


def merge(df, name, test=False):
    oof, ocol, tst, tcol, israw = REG[name]
    path, col = (tst, tcol) if test else (oof, ocol)
    if not os.path.exists(path):
        return df, False
    d = pd.read_parquet(path)[["contact_id", col]].rename(columns={col: name})
    d = d.dropna(subset=["contact_id"]).drop_duplicates("contact_id")
    return df.merge(d, on="contact_id", how="left"), True


def blend_scores(df, weights):
    n = len(df); num = np.zeros(n); den = np.zeros(n)
    for s, w in weights.items():
        z = df[s].values.astype(np.float64) if s in RAW_LOGIT else logit(df[s].values)
        m = np.isfinite(z)
        num[m] += w * z[m]; den[m] += w
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def greedy(df, y, names, keys, wss, max_add=5, min_gain=2e-4):
    sm = make_smoother(df, np.ones(len(df), bool), keys)
    best = None
    for s in names:
        if df[s].notna().mean() < 1e-4:
            continue
        for ws in wss:
            sc = sm(logit(df[s].values) if s not in RAW_LOGIT else df[s].values, ws)
            _, m = optimize_thresholds([Sub(y, sc, s)])
            if best is None or m > best[0]:
                best = (m, {s: 1.0}, ws, sc)
    print("  base", round(best[0], 5), best[1], "ws", best[2], flush=True)
    added = 0
    improved = True
    while improved and added < max_add:
        improved = False
        for s in names:
            if s in best[1] or df[s].notna().mean() < 1e-4:
                continue
            loc = None
            for w in [0.25, 0.5, 0.8, 1.2]:
                cand = dict(best[1]); cand[s] = w
                z = blend_scores(df, cand)
                for ws in wss:
                    sc = sm(z, ws)
                    _, m = optimize_thresholds([Sub(y, sc, s)])
                    if m > best[0] + min_gain and (loc is None or m > loc[0]):
                        loc = (m, cand, ws, sc)
            if loc is not None:
                best = loc; improved = True; added += 1
                print("  +", s, loc[1][s], "ws", loc[2], "->", round(loc[0], 5), best[1], flush=True)
    return best


def nested_eval(folds_pp, y_pp, spp, folds_gr, y_gr, sgr, rate=False):
    TP = FP = P = N = 0
    for f in np.unique(folds_pp):
        for y, sc, folds in [(y_pp, spp, folds_pp), (y_gr, sgr, folds_gr)]:
            trm = folds != f; vam = folds == f
            sb_tr = Sub(y[trm], sc[trm], "t")
            ks, _ = optimize_thresholds([sb_tr])
            k = ks[0]
            th = sb_tr.th[k] if k >= 0 else np.inf
            if rate:
                tp, fp = sb_tr.counts(k)
                r = (tp + fp) / max(1, len(sb_tr.y))
                th = np.quantile(sc[vam], 1 - r) if vam.sum() > 0 else np.inf
            yv = y[vam]; sv = sc[vam]
            TP += int(((sv >= th) & (yv == 1)).sum()); FP += int(((sv >= th) & (yv == 0)).sum())
            P += int(yv.sum()); N += int((1 - yv).sum())
    FN = P - TP; TN = N - FP
    d = np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return 0.0 if d == 0 else (TP * TN - FP * FN) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="final_solution")
    args = ap.parse_args()
    tr = base_train(); te = base_test()
    for name in REG:
        tr, ok = merge(tr, name, False)
        te, ok2 = merge(te, name, True)
        if not ok or not ok2:
            print("missing", name)
    pp = ~tr.is_ground.values; gr = tr.is_ground.values
    pp_y = tr.contact.values[pp].astype(np.int8); gr_y = tr.contact.values[gr].astype(np.int8)
    pp_keys = ["game_play", "p1", "p2s"]; gr_keys = ["game_play", "p1"]
    tr_pp = tr[pp].reset_index(drop=True); tr_gr = tr[gr].reset_index(drop=True)
    pp_names = [s for s in PP_SRC if s in tr.columns and tr.loc[pp, s].notna().mean() > 1e-4]
    gr_names = [s for s in GR_SRC if s in tr.columns and tr.loc[gr, s].notna().mean() > 1e-4]
    print("PP", pp_names); print("GR", gr_names, flush=True)
    bpp = greedy(tr_pp, pp_y, pp_names, pp_keys, [0, 1, 2])
    bgr = greedy(tr_gr, gr_y, gr_names, gr_keys, [0, 2, 4, 6])
    subs = [Sub(pp_y, bpp[3], "pp"), Sub(gr_y, bgr[3], "gr")]
    ks, M = optimize_thresholds(subs)
    # target positive rates at OOF-optimal thresholds
    tp_pp, fp_pp = subs[0].counts(ks[0]); tp_gr, fp_gr = subs[1].counts(ks[1])
    rate_pp = (tp_pp + fp_pp) / len(pp_y); rate_gr = (tp_gr + fp_gr) / len(gr_y)
    folds_pp = tr.fold.values[pp]; folds_gr = tr.fold.values[gr]
    n_fix = nested_eval(folds_pp, pp_y, bpp[3], folds_gr, gr_y, bgr[3], rate=False)
    n_rate = nested_eval(folds_pp, pp_y, bpp[3], folds_gr, gr_y, bgr[3], rate=True)
    print("OOF", round(M, 5), "pp", round(bpp[0], 5), "gr", round(bgr[0], 5),
          "rates", round(rate_pp, 5), round(rate_gr, 5), "nested_fixed", round(n_fix, 5),
          "nested_rate", round(n_rate, 5), flush=True)
    use_rate = n_rate >= n_fix
    # test scores
    te_pp = ~te.is_ground.values; te_gr = te.is_ground.values
    zpp = blend_scores(te.loc[te_pp].reset_index(drop=True), bpp[1])
    spp = make_smoother(te, te_pp, pp_keys)(zpp, bpp[2])
    zgr = blend_scores(te.loc[te_gr].reset_index(drop=True), bgr[1])
    sgr = make_smoother(te, te_gr, gr_keys)(zgr, bgr[2])
    pred = np.zeros(len(te), np.int8)
    if use_rate:
        th_pp = np.quantile(spp, 1 - rate_pp); th_gr = np.quantile(sgr, 1 - rate_gr)
    else:
        th_pp = float(subs[0].th[ks[0]]); th_gr = float(subs[1].th[ks[1]])
    pred[te_pp] = (spp >= th_pp).astype(np.int8)
    pred[te_gr] = (sgr >= th_gr).astype(np.int8)
    sub = pd.DataFrame({"contact_id": te.contact_id, "contact": pred})
    os.makedirs(args.outdir, exist_ok=True)
    sub.to_csv(f"{args.outdir}/submission.csv", index=False)
    # save config + OOF/test scores
    used = set(bpp[1]) | set(bgr[1])
    cfg_sources = {}
    for u in sorted(used):
        oof, ocol, tst, tcol, israw = REG[u]
        cfg_sources[u] = {"path": tst, "column": tcol}
    cfg = {"sources": cfg_sources, "raw_logit": [u for u in used if u in RAW_LOGIT],
           "keys": {"pp": pp_keys, "gr": gr_keys},
           "pp": {"weights": bpp[1], "ws": bpp[2], "th": float(th_pp)},
           "gr": {"weights": bgr[1], "ws": bgr[2], "th": float(th_gr)},
           "rate_match": bool(use_rate), "rates": {"pp": float(rate_pp), "gr": float(rate_gr)}}
    json.dump(cfg, open(f"{OUT}/final_blend_config.json", "w"), indent=2)
    oof_score = np.full(len(tr), np.nan); oof_score[pp] = bpp[3]; oof_score[gr] = bgr[3]
    pd.DataFrame({"contact_id": tr.contact_id.values, "is_ground": tr.is_ground.values,
                  "fold": tr.fold.values, "contact": tr.contact.values,
                  "score": oof_score}).to_parquet(f"{OUT}/oof_final_blend.parquet", index=False)
    te_score = np.full(len(te), np.nan); te_score[te_pp] = spp; te_score[te_gr] = sgr
    pd.DataFrame({"contact_id": te.contact_id.values, "is_ground": te.is_ground.values,
                  "score": te_score}).to_parquet(f"{OUT}/test_final_probs.parquet", index=False)
    print("wrote submission", sub.shape, "pos", sub.contact.mean(),
          "pp", pred[te_pp].mean(), "gr", pred[te_gr].mean(), "use_rate", use_rate)


if __name__ == "__main__":
    main()
