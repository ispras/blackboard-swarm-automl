"""Reproduce final_solution/submission.csv from the frozen source prediction files.

Usage:
    pyenv shell sis-black
    python final_solution/code/make_submission.py

Reads `final_solution/final_config.json` (source paths + blend weights + smoothing
windows + logit thresholds) and writes `final_solution/submission.csv` in sample order.

All source predictions were produced by 5-fold GroupKFold(game_play) models trained
only on the provided training split (`data/train_*`). Test-time inputs are the
provided mock-test files in `data/test*`. No test labels were used.
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTDIR = os.path.join(ROOT, "final_solution")
CFG = json.load(open(os.path.join(OUTDIR, "final_config.json")))
RAW_LOGIT = set(CFG.get("raw_logit", []))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def base_test():
    sub = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"))
    parts = sub.contact_id.str.rsplit("_", n=3, expand=True)
    out = pd.DataFrame({"contact_id": sub.contact_id, "game_play": parts[0],
                        "step": parts[1].astype("int32"), "p1": parts[2].astype("int32"),
                        "p2s": parts[3].astype(str)})
    out["is_ground"] = out.p2s == "G"
    return out


def load_source(df, path, col, name):
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    d = pd.read_parquet(path)
    d = d[["contact_id", col]].rename(columns={col: name})
    d = d.dropna(subset=["contact_id"]).drop_duplicates("contact_id")
    return df.merge(d, on="contact_id", how="left")


def blend_scores(df, weights):
    n = len(df)
    num = np.zeros(n)
    den = np.zeros(n)
    for s, w in weights.items():
        z = df[s].values.astype(np.float64) if s in RAW_LOGIT else logit(df[s].values)
        m = np.isfinite(z)
        num[m] += w * z[m]
        den[m] += w
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def make_smoother(df, mask, keys):
    sub = df.loc[mask, list(keys) + ["step"]]
    code = pd.factorize(pd.MultiIndex.from_arrays([sub[k].values for k in keys]))[0]
    step = sub["step"].values
    order = np.lexsort((step, code))
    inv = np.empty(len(order), np.int64)
    inv[order] = np.arange(len(order))
    cs = code[order]

    def f(score, ws):
        if ws is None or ws <= 0:
            return np.asarray(score, np.float64)
        ss = np.asarray(score, np.float64)[order]
        roll = pd.Series(ss).groupby(cs, sort=False).rolling(
            2 * ws + 1, center=True, min_periods=1).mean().values
        return roll[inv]
    return f


def main():
    df = base_test()
    for name, spec in CFG["sources"].items():
        df = load_source(df, spec["path"], spec["column"], name)
    pp = ~df.is_ground.values
    gr = df.is_ground.values
    pp_keys = CFG["keys"]["pp"]
    gr_keys = CFG["keys"]["gr"]
    cfg_pp = CFG["pp"]
    cfg_gr = CFG["gr"]

    zpp = blend_scores(df[pp].reset_index(drop=True), cfg_pp["weights"])
    spp = make_smoother(df, pp, pp_keys)(zpp, cfg_pp["ws"])
    zgr = blend_scores(df[gr].reset_index(drop=True), cfg_gr["weights"])
    sgr = make_smoother(df, gr, gr_keys)(zgr, cfg_gr["ws"])

    if CFG.get("rate_match"):
        th_pp = float(np.quantile(spp, 1 - CFG["rates"]["pp"]))
        th_gr = float(np.quantile(sgr, 1 - CFG["rates"]["gr"]))
    else:
        th_pp = float(cfg_pp["th"]); th_gr = float(cfg_gr["th"])
    pred = np.zeros(len(df), np.int8)
    pred[pp] = (spp >= th_pp).astype(np.int8)
    pred[gr] = (sgr >= th_gr).astype(np.int8)
    out = pd.DataFrame({"contact_id": df.contact_id, "contact": pred})
    out.to_csv(os.path.join(OUTDIR, "submission.csv"), index=False)
    print("wrote", os.path.join(OUTDIR, "submission.csv"), out.shape,
          "pos_rate", float(out.contact.mean()))


if __name__ == "__main__":
    main()
