import numpy as np, pandas as pd, os, sys, json, argparse
OUT = "workspaces/alkaid_psi"
K = 500


# ---------------- base tables ----------------
def base_train():
    lab = pd.read_csv("data/train_labels.csv",
                      usecols=["contact_id", "game_play", "step", "nfl_player_id_1", "nfl_player_id_2", "contact"])
    lab["is_ground"] = lab.nfl_player_id_2.astype(str) == "G"
    fold = pd.read_csv(f"{OUT}/cv_folds.csv")
    lab = lab.merge(fold, on="game_play", how="left")
    return lab[["contact_id", "game_play", "step", "contact", "is_ground", "fold"]]


def base_test():
    sub = pd.read_csv("data/sample_submission.csv")
    parts = sub.contact_id.str.rsplit("_", n=3, expand=True)
    out = pd.DataFrame({"contact_id": sub.contact_id, "game_play": parts[0],
                        "step": parts[1].astype("int32"), "p2s": parts[3].astype(str)})
    out["is_ground"] = out.p2s == "G"
    return out


def load_sources(tr, te, sources):
    names = []
    for name, oof, test in sources:
        if not (os.path.exists(oof) and os.path.exists(test)):
            print("skip missing source", name)
            continue
        o = pd.read_parquet(oof)[["contact_id", "pred"]].rename(columns={"pred": name})
        t = pd.read_parquet(test)[["contact_id", "pred"]].rename(columns={"pred": name})
        tr = tr.merge(o, on="contact_id", how="left")
        te = te.merge(t, on="contact_id", how="left")
        names.append(name)
        print("loaded", name, "train cov", tr[name].notna().mean(), "test cov", te[name].notna().mean())
    return tr, te, names


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------- thresholds / MCC ----------------
class Sub:
    def __init__(self, y, s, name):
        self.name = name
        self.y = y.astype(np.int8)
        self.s = s.astype(np.float64)
        o = np.argsort(-self.s, kind="stable")
        ys = self.y[o].astype(np.int64)
        self.ss = self.s[o]
        self.tp = np.cumsum(ys)
        self.fp = np.cumsum(1 - ys)
        self.P = int(self.y.astype(np.int64).sum()); self.N = int(len(self.y) - self.P)
        n = len(self.s)
        self.ridx = np.unique(np.linspace(0, n - 1, K).astype(int))
        # add sentinel "none positive" as index -1
        self.th = self.ss[self.ridx]

    def counts(self, k):
        if k < 0:
            return 0, 0
        return int(self.tp[self.ridx[k]]), int(self.fp[self.ridx[k]])


def global_mcc(subs, ks):
    TP = FP = 0
    P = N = 0
    for sb, k in zip(subs, ks):
        tp, fp = sb.counts(k)
        TP += tp; FP += fp; P += sb.P; N += sb.N
    FN = P - TP; TN = N - FP
    den = np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return 0.0 if den == 0 else (TP * TN - FP * FN) / den


def optimize_thresholds(subs, rounds=4):
    ks = [None] * len(subs)
    for i, sb in enumerate(subs):
        # init ~ threshold that maximizes own mcc
        best = (-2, 0)
        for k in range(len(sb.ridx)):
            m = global_mcc([sb], [k])
            if m > best[0]:
                best = (m, k)
        ks[i] = best[1]
    for _ in range(rounds):
        changed = False
        for i in range(len(subs)):
            best = (-2, ks[i])
            for k in range(-1, len(subs[i].ridx)):
                cur = list(ks); cur[i] = k
                m = global_mcc(subs, cur)
                if m > best[0]:
                    best = (m, k)
            if best[1] != ks[i]:
                ks[i] = best[1]; changed = True
        if not changed:
            break
    return ks, global_mcc(subs, ks)


def greedy_blend(sub_df, cols, y, subs_all, sub_idx, name, thresh_ks_other, rounds=3):
    """Greedy add logit-weighted sources for one subset; returns blended score + weights."""
    avail = [c for c in cols if c in sub_df.columns and sub_df[c].notna().any()]
    if not avail:
        return None, {}
    X = {c: logit(sub_df[c].fillna(sub_df[c].median()).values) for c in avail}
    # start with best single source by own mcc
    base = None; bestm = -2
    for c in avail:
        sb = Sub(y, X[c], name + ":" + c)
        _, mval = optimize_thresholds([sb])
        if mval > bestm:
            bestm, base = mval, c
    weights = {base: 1.0}
    score = X[base].copy()

    def build_score(w):
        z = np.zeros_like(score); tot = 0.0
        for c, wi in w.items():
            z += wi * X[c]; tot += wi
        return z / max(tot, 1e-9)

    def eval_with(score_new):
        sb = Sub(y, score_new, name)
        _, m = optimize_thresholds([sb])
        return m
    # greedy: try adding each other source with weights
    for c in avail:
        if c == base:
            continue
        improved = False
        for w in [0.15, 0.3, 0.5, 0.8, 1.2]:
            w2 = dict(weights); w2[c] = w
            m = eval_with(build_score(w2))
            if m > bestm + 1e-5:
                bestm = m; weights = w2; improved = True
        if improved:
            score = build_score(weights)
    return weights, float(bestm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="v1")
    ap.add_argument("--outdir", default="final_solution")
    args = ap.parse_args()
    tr = base_train(); te = base_test()
    sources = [
        ("can_pp", "workspaces/canopus_pi/preds/oof_pp_v1.parquet", "workspaces/canopus_pi/preds/test_pp_v1.parquet"),
        ("can_gr", "workspaces/canopus_pi/preds/oof_ground_v1.parquet", "workspaces/canopus_pi/preds/test_ground_v1.parquet"),
        ("cnn_pp", f"{OUT}/oof_cnn_cand_{args.suffix}.parquet", f"{OUT}/test_cnn_cand_{args.suffix}.parquet"),
    ]
    # optional capella (all rows) and alkaid ground
    sources += [
        ("cap", "workspaces/capella_pi/oof_lgbm.parquet", "workspaces/capella_pi/test_lgbm.parquet"),
        ("cnn_gr", f"{OUT}/oof_ground_cnn.parquet", f"{OUT}/test_ground_cnn.parquet"),
    ]
    tr, te, names = load_sources(tr, te, sources)
    tr.to_parquet(f"{OUT}/blend_train.parquet", index=False)
    te.to_parquet(f"{OUT}/blend_test.parquet", index=False)

    pp_cand = (~tr.is_ground) & tr.cnn_pp.notna()
    pp_non = (~tr.is_ground) & tr.cnn_pp.isna()
    gr = tr.is_ground
    pp_cols = [c for c in ["can_pp", "cap", "cnn_pp"] if c in tr.columns]
    gr_cols = [c for c in ["can_gr", "cap", "cnn_gr"] if c in tr.columns]
    masks = {"pp_cand": pp_cand, "pp_non": pp_non, "gr": gr}
    print("subset sizes", {k: int(v.sum()) for k, v in masks.items()})

    results = {}
    # --- pp_cand blend (global objective not needed yet; optimize within subset) ---
    d = tr.loc[pp_cand]
    w_pp, m_pp = greedy_blend(d, pp_cols, d.contact.values, None, None, "pp_cand", None)
    print("pp_cand weights", w_pp, "mcc_subset", m_pp)
    # --- pp_non blend ---
    d = tr.loc[pp_non]
    w_pn, m_pn = greedy_blend(d, [c for c in pp_cols if c != "cnn_pp"] or pp_cols,
                              d.contact.values, None, None, "pp_non", None)
    print("pp_non weights", w_pn, "mcc_subset", m_pn)
    # --- ground blend ---
    d = tr.loc[gr]
    w_gr, m_gr = greedy_blend(d, gr_cols, d.contact.values, None, None, "gr", None)
    print("ground weights", w_gr, "mcc_subset", m_gr)

    def apply_blend(df, weights, cols):
        X = {c: logit(df[c].fillna(df[c].median()).values) for c in weights if c in df.columns}
        z = np.zeros(len(df)); tot = 0
        for c, wi in weights.items():
            if c in X:
                z += wi * X[c]; tot += wi
        return z / max(tot, 1e-9)

    s_pc = apply_blend(tr.loc[pp_cand], w_pp, pp_cols)
    s_pn = apply_blend(tr.loc[pp_non], w_pn, pp_cols)
    s_gr = apply_blend(tr.loc[gr], w_gr, gr_cols)
    subs = [Sub(tr.loc[pp_cand].contact.values, s_pc, "pp_cand"),
            Sub(tr.loc[pp_non].contact.values, s_pn, "pp_non"),
            Sub(tr.loc[gr].contact.values, s_gr, "gr")]
    ks, M = optimize_thresholds(subs)
    print("FINAL OOF MCC", M)
    th = {}
    for sb, k, name in zip(subs, ks, ["pp_cand", "pp_non", "gr"]):
        th[name] = float(sb.th[k]) if k >= 0 else float("inf")
        tp, fp = sb.counts(k)
        print(name, "th_logit", th[name], "tp", tp, "fp", fp, "P", sb.P, "N", sb.N)

    # ---- test ----
    te_pp_cand = (~te.is_ground) & te.cnn_pp.notna()
    te_pp_non = (~te.is_ground) & te.cnn_pp.isna()
    te_gr = te.is_ground
    pred = np.zeros(len(te), np.float64)
    for mask, weights, cols, name in [
        (te_pp_cand, w_pp, pp_cols, "pp_cand"),
        (te_pp_non, w_pn, pp_cols, "pp_non"),
        (te_gr, w_gr, gr_cols, "gr")]:
        if mask.sum() == 0:
            continue
        s = apply_blend(te.loc[mask], weights, cols)
        pred[mask.values] = s
        print("test", name, int(mask.sum()), "mean", float(s.mean()), "th", th[name])
    # map logit threshold -> probability for reporting; predict via logit
    lab = np.zeros(len(te), np.int8)
    for mask, name in [(te_pp_cand, "pp_cand"), (te_pp_non, "pp_non"), (te_gr, "gr")]:
        if mask.sum() == 0:
            continue
        s = pred[mask.values]
        lab[mask.values] = (s >= th[name]).astype(np.int8)
    sub = pd.DataFrame({"contact_id": te.contact_id, "contact": lab})
    os.makedirs(args.outdir, exist_ok=True)
    sub.to_csv(f"{args.outdir}/submission.csv", index=False)
    print("wrote", f"{args.outdir}/submission.csv", sub.shape, "pos rate", sub.contact.mean())
    json.dump({"oof_mcc": M, "weights": {"pp_cand": w_pp, "pp_non": w_pn, "gr": w_gr},
               "thresholds_logit": th, "sources": names,
               "subset_mcc": {"pp_cand": m_pp, "pp_non": m_pn, "gr": m_gr}},
              open(f"{OUT}/blend_report.json", "w"), indent=2)
    print("saved report")


if __name__ == "__main__":
    main()
