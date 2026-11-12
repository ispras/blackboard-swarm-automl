import numpy as np, pandas as pd, os, sys, json, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blend_final import Sub, optimize_thresholds, logit, global_mcc

OUT = "workspaces/alkaid_psi"


def base_train():
    lab = pd.read_csv("data/train_labels.csv",
                      usecols=["contact_id", "game_play", "step", "nfl_player_id_1", "nfl_player_id_2", "contact"])
    lab["p1"] = lab.nfl_player_id_1.astype("int32")
    lab["p2s"] = lab.nfl_player_id_2.astype(str)
    lab["is_ground"] = lab.p2s == "G"
    fold = pd.read_csv(f"{OUT}/cv_folds.csv")
    lab = lab.merge(fold, on="game_play", how="left")
    return lab[["contact_id", "game_play", "step", "p1", "p2s", "contact", "is_ground", "fold"]]


def base_test():
    sub = pd.read_csv("data/sample_submission.csv")
    parts = sub.contact_id.str.rsplit("_", n=3, expand=True)
    out = pd.DataFrame({"contact_id": sub.contact_id, "game_play": parts[0],
                        "step": parts[1].astype("int32"), "p1": parts[2].astype("int32"), "p2s": parts[3].astype(str)})
    out["is_ground"] = out.p2s == "G"
    return out


def merge_src(df, path, col, keys=("contact_id",)):
    if not os.path.exists(path):
        print("missing", path); return df
    d = pd.read_parquet(path)
    keep = [k for k in keys if k in d.columns]
    d = d[keep + ["pred"]].rename(columns={"pred": col})
    if "contact_id" in keys:
        d = d.drop_duplicates("contact_id")
        df = df.merge(d, on="contact_id", how="left")
    print("merged", col, path, "cov", df[col].notna().mean())
    return df


def make_smoother(df, mask, keys):
    sub = df.loc[mask, list(keys) + ["step"]]
    code = pd.factorize(pd.MultiIndex.from_arrays([sub[k].values for k in keys]))[0]
    step = sub["step"].values
    order = np.lexsort((step, code))
    inv = np.empty(len(order), np.int64); inv[order] = np.arange(len(order))
    cs = code[order]

    def f(score, ws):
        if ws is None or ws <= 0:
            return np.asarray(score, np.float64)
        ss = np.asarray(score, np.float64)[order]
        roll = pd.Series(ss).groupby(cs, sort=False).rolling(2 * ws + 1, center=True, min_periods=1).mean().values
        return roll[inv]
    return f


def eval_cfg(pp_y, pp_z, gr_y, gr_z):
    subs = [Sub(pp_y, pp_z, "pp"), Sub(gr_y, gr_z, "gr")]
    ks, M = optimize_thresholds(subs)
    return M, ks, subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn_tag", default="v2m")
    ap.add_argument("--cnn_gr", default="v1")
    ap.add_argument("--outdir", default="final_solution")
    ap.add_argument("--can", default="v2")
    args = ap.parse_args()
    tr = base_train(); te = base_test()
    tr = merge_src(tr, f"workspaces/canopus_pi/preds/oof_pp_{args.can}.parquet", "can_pp")
    tr = merge_src(tr, f"workspaces/canopus_pi/preds/oof_ground_{args.can}.parquet", "can_gr")
    if args.can != "v1":
        tr = merge_src(tr, "workspaces/canopus_pi/preds/oof_pp_v1.parquet", "can_pp1")
        tr = merge_src(tr, "workspaces/canopus_pi/preds/oof_ground_v1.parquet", "can_gr1")
    tr = merge_src(tr, f"{OUT}/oof_cnn_cand_{args.cnn_tag}.parquet", "cnn_pp")
    tr = merge_src(tr, f"{OUT}/oof_ground_cnn_{args.cnn_gr}.parquet", "cnn_gr")
    tr = merge_src(tr, "workspaces/capella_pi/oof_lgbm.parquet", "cap")
    te = merge_src(te, f"workspaces/canopus_pi/preds/test_pp_{args.can}.parquet", "can_pp")
    te = merge_src(te, f"workspaces/canopus_pi/preds/test_ground_{args.can}.parquet", "can_gr")
    if args.can != "v1":
        te = merge_src(te, "workspaces/canopus_pi/preds/test_pp_v1.parquet", "can_pp1")
        te = merge_src(te, "workspaces/canopus_pi/preds/test_ground_v1.parquet", "can_gr1")
    te = merge_src(te, f"{OUT}/test_cnn_cand_{args.cnn_tag}.parquet", "cnn_pp")
    te = merge_src(te, f"{OUT}/test_ground_cnn_{args.cnn_gr}.parquet", "cnn_gr")
    te = merge_src(te, "workspaces/capella_pi/test_lgbm.parquet", "cap")
    tr.to_parquet(f"{OUT}/blend2_train.parquet", index=False)
    te.to_parquet(f"{OUT}/blend2_test.parquet", index=False)

    has_cap = "cap" in tr.columns and tr.cap.notna().any()
    has_cnn_gr = "cnn_gr" in tr.columns and tr.cnn_gr.notna().any()
    pp = ~tr.is_ground.values
    gr = tr.is_ground.values
    print("pp", pp.sum(), "gr", gr.sum(), "has_cap", has_cap, "has_cnn_gr", has_cnn_gr)

    zcan = logit(tr.can_pp.values)
    if "can_pp1" in tr.columns:
        z0 = logit(tr.can_pp1.values)
        zcan = np.where(np.isfinite(z0), 0.5 * (zcan + z0), zcan)
    zcnn = logit(tr.cnn_pp.fillna(0.5).values) if "cnn_pp" in tr.columns else np.zeros(len(tr))
    zcap = logit(tr.cap.fillna(0.5).values) if has_cap else np.zeros(len(tr))
    has_cnn = tr.cnn_pp.notna().values if "cnn_pp" in tr.columns else np.zeros(len(tr), bool)
    has_capp = tr.cap.notna().values if has_cap else np.zeros(len(tr), bool)

    def pp_score(wc, wcap):
        num = zcan.copy(); den = np.ones(len(tr))
        cnn_w = wc * has_cnn
        num += cnn_w * zcnn; den += cnn_w
        cap_w = wcap * has_capp
        num += cap_w * zcap; den += cap_w
        return num / den

    pp_y = tr.contact.values[pp].astype(np.int8)
    gr_y = tr.contact.values[gr].astype(np.int8)
    zgr_can = logit(tr.can_gr.fillna(0.5).values)
    if "can_gr1" in tr.columns:
        z0 = logit(tr.can_gr1.fillna(0.5).values)
        zgr_can = np.where(np.isfinite(z0), 0.5 * (zgr_can + z0), zgr_can)
    zgr_cnn = logit(tr.cnn_gr.fillna(0.5).values) if has_cnn_gr else np.zeros(len(tr))
    has_grcnn = tr.cnn_gr.notna().values if has_cnn_gr else np.zeros(len(tr), bool)

    def gr_score(wg):
        num = zgr_can.copy(); den = np.ones(len(tr))
        num += wg * has_grcnn * zgr_cnn; den += wg * has_grcnn
        return num / den

    pp_keys = ["game_play", "p1", "p2s"]
    gr_keys = ["game_play", "p1"]
    best = None
    pp_smooth = make_smoother(tr, pp, pp_keys)
    gr_smooth = make_smoother(tr, gr, gr_keys)
    # stage 1: pp weights + smoothing, ground baseline
    gr_base = gr_score(0.0)[gr]
    for wc in [0.0, 0.2, 0.35, 0.5, 0.7, 1.0]:
        for wcap in ([0.0] if not has_cap else [0.0, 0.25, 0.5]):
            base = pp_score(wc, wcap)
            for ws in [0, 1, 2]:
                sm = pp_smooth(base[pp], ws)
                M, ks, subs = eval_cfg(pp_y, sm, gr_y, gr_base)
                print(f"pp wc={wc} wcap={wcap} ws={ws} -> MCC {M:.5f}")
                if best is None or M > best[0]:
                    best = (M, wc, wcap, ws, None, None)
    M, wc, wcap, ws, _, _ = best
    print("BEST PP", best[:4])
    pp_best = pp_smooth(pp_score(wc, wcap)[pp], ws)
    # stage 2: ground weight + smoothing
    gbest = (M, 0.0, 0)
    for wg in [0.0, 0.15, 0.3, 0.5, 0.8]:
        if not has_cnn_gr and wg > 0:
            break
        gbase = gr_score(wg)[gr]
        for wgs in [0, 2, 4, 6, 8]:
            sm = gr_smooth(gbase, wgs)
            M2, ks, subs = eval_cfg(pp_y, pp_best, gr_y, sm)
            print(f"gr wg={wg} ws={wgs} -> MCC {M2:.5f}")
            if M2 > gbest[0]:
                gbest = (M2, wg, wgs)
    M2, wg, wgs = gbest
    print("BEST GR", gbest)
    pp_best = pp_smooth(pp_score(wc, wcap)[pp], ws)
    gr_best = gr_smooth(gr_score(wg)[gr], wgs)
    M2, ks, subs = eval_cfg(pp_y, pp_best, gr_y, gr_best)
    print("FINAL OOF MCC", M2)
    tp_pp, fp_pp = subs[0].counts(ks[0]); tp_gr, fp_gr = subs[1].counts(ks[1])
    print("pp th", subs[0].th[ks[0]] if ks[0] >= 0 else "inf", "tp", tp_pp, "fp", fp_pp)
    print("gr th", subs[1].th[ks[1]] if ks[1] >= 0 else "inf", "tp", tp_gr, "fp", fp_gr)

    # ---- test ----
    te_pp = ~te.is_ground.values; te_gr = te.is_ground.values
    zcan_t = logit(te.can_pp.values)
    if "can_pp1" in te.columns:
        z0 = logit(te.can_pp1.values)
        zcan_t = np.where(np.isfinite(z0), 0.5 * (zcan_t + z0), zcan_t)
    zcnn_t = logit(te.cnn_pp.fillna(0.5).values)
    has_cnn_t = te.cnn_pp.notna().values
    num = zcan_t.copy(); den = np.ones(len(te))
    num += wc * has_cnn_t * zcnn_t; den += wc * has_cnn_t
    if has_cap:
        zcap_t = logit(te.cap.fillna(0.5).values); has_capt = te.cap.notna().values
        num += wcap * has_capt * zcap_t; den += wcap * has_capt
    pps = num / den
    pps = make_smoother(te, te_pp, pp_keys)(pps[te_pp], ws)
    zcan_g = logit(te.can_gr.fillna(0.5).values)
    if "can_gr1" in te.columns:
        z0 = logit(te.can_gr1.fillna(0.5).values)
        zcan_g = np.where(np.isfinite(z0), 0.5 * (zcan_g + z0), zcan_g)
    numg = zcan_g.copy(); deng = np.ones(len(te))
    if has_cnn_gr:
        zcnn_g = logit(te.cnn_gr.fillna(0.5).values); has_g = te.cnn_gr.notna().values
        numg += wg * has_g * zcnn_g; deng += wg * has_g
    grs = numg / deng
    grs = make_smoother(te, te_gr, gr_keys)(grs[te_gr], wgs)
    pred = np.zeros(len(te), np.int8)
    pred[te_pp] = (pps >= subs[0].th[ks[0]]).astype(np.int8)
    pred[te_gr] = (grs >= subs[1].th[ks[1]]).astype(np.int8)
    sub = pd.DataFrame({"contact_id": te.contact_id, "contact": pred})
    os.makedirs(args.outdir, exist_ok=True)
    sub.to_csv(f"{args.outdir}/submission.csv", index=False)
    print("wrote submission", sub.shape, "pos rate", sub.contact.mean(), "pp pos", pred[te_pp].mean(), "gr pos", pred[te_gr].mean())
    json.dump({"oof_mcc": M2, "wc": wc, "wcap": wcap, "ws": ws, "wg": wg, "wgs": wgs,
               "th_pp": float(subs[0].th[ks[0]]), "th_gr": float(subs[1].th[ks[1]]),
               "has_cap": bool(has_cap), "has_cnn_gr": bool(has_cnn_gr),
               "cnn_tag": args.cnn_tag, "cnn_gr": args.cnn_gr, "can": args.can},
              open(f"{OUT}/blend2_report.json", "w"), indent=2)


if __name__ == "__main__":
    main()
