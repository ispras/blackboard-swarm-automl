import time, json, warnings, re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss
import lightgbm as lgb
warnings.filterwarnings("ignore")

WS = "workspaces/arcturus_alpha"
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = np.load(f"{WS}/y.npy")
folds = np.load(f"{WS}/folds.npy")
NF = 5


def char_feats(texts):
    rows = []
    for t in texts:
        n = len(t)
        words = t.split()
        nw = max(len(words), 1)
        wl = [len(w) for w in words]
        low = t.lower()
        rows.append([
            n, nw, np.mean(wl), np.std(wl) if wl else 0, max(wl) if wl else 0,
            len(set(w.lower() for w in words)) / nw,
            t.count(","), t.count("."), t.count(";"), t.count(":"),
            t.count("!"), t.count("?"), t.count('"'), t.count("'"),
            t.count("-"), t.count("("), t.count(")"),
            sum(c.isupper() for c in t) / max(n, 1),
            sum(c.isdigit() for c in t) / max(n, 1),
            sum(c == " " for c in t) / max(n, 1),
            sum(c in "aeiou" for c in low) / max(n, 1),
            np.mean([len(w) for w in words if w]) if words else 0,
            len(re.findall(r"\bthe\b", low)), len(re.findall(r"\bof\b", low)),
            len(re.findall(r"\band\b", low)), len(re.findall(r"\bto\b", low)),
            len(re.findall(r"\ba\b", low)), len(re.findall(r"\bwhich\b", low)),
            len(re.findall(r"\bthat\b", low)), len(re.findall(r"\bI\b", t)),
            len(re.findall(r"\bmy\b", low)), len(re.findall(r"\bhis\b", low)),
            len(re.findall(r"\bher\b", low)), len(re.findall(r"\byou\b", low)),
            t.count(",") / nw, t.count(";") / nw,
            len(re.findall(r"[A-Z][a-z]+", t)) / nw,
            len(re.findall(r"\b\w{11,}\b", low)),
            low.count("the "), low.count("ing"), low.count("ed "),
            len(re.findall(r"\bnot\b", low)), len(re.findall(r"\bno\b", low)),
        ])
    return np.array(rows, dtype=np.float32)


def tfidf3(trT, vaT, teT):
    vs = [
        TfidfVectorizer(analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1, 2), min_df=1,
                        lowercase=False, sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                        lowercase=False, sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1,
                        lowercase=False, sublinear_tf=True, strip_accents="unicode"),
    ]
    tr, va, te = [], [], []
    for v in vs:
        v.fit(trT)
        tr.append(v.transform(trT)); va.append(v.transform(vaT)); te.append(v.transform(teT))
    return hstack(tr).tocsr(), hstack(va).tocsr(), hstack(te).tocsr()


def run_lgbm(name, use_svd, use_hand, n_est=700, lr=0.05, leaves=31):
    oof = np.zeros((len(y), 3)); testp = np.zeros((len(test), 3))
    txt = train["text"].values; ttxt = test["text"].values
    Htr_all = char_feats(txt); Hte_all = char_feats(ttxt)
    t0 = time.time()
    for f in range(NF):
        tr, va = folds != f, folds == f
        mats_tr, mats_va, mats_te = [], [], []
        if use_svd:
            Xtr, Xva, Xte = tfidf3(txt[tr], txt[va], ttxt)
            svd = TruncatedSVD(n_components=200, random_state=42)
            mats_tr.append(svd.fit_transform(Xtr)); mats_va.append(svd.transform(Xva)); mats_te.append(svd.transform(Xte))
        if use_hand:
            mats_tr.append(Htr_all[tr]); mats_va.append(Htr_all[va]); mats_te.append(Hte_all)
        Xtr = np.hstack(mats_tr); Xva = np.hstack(mats_va); Xte = np.hstack(mats_te)
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=n_est,
                                 learning_rate=lr, num_leaves=leaves, feature_fraction=0.7,
                                 bagging_fraction=0.8, bagging_freq=1, min_child_samples=20,
                                 subsample=0.8, colsample_bytree=0.7, n_jobs=8, random_state=42, verbose=-1)
        clf.fit(Xtr, y[tr])
        oof[va] = clf.predict_proba(Xva)
        testp += clf.predict_proba(Xte) / NF
    ll = log_loss(y, oof, labels=[0, 1, 2])
    np.save(f"{WS}/oof_{name}.npy", oof); np.save(f"{WS}/test_{name}.npy", testp)
    print(f"{name:24s} OOF={ll:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return ll


if __name__ == "__main__":
    res = {}
    res["lgbm_hand"] = run_lgbm("lgbm_hand", False, True)
    res["lgbm_svd"] = run_lgbm("lgbm_svd", True, False)
    res["lgbm_svd_hand"] = run_lgbm("lgbm_svd_hand", True, True)
    print(json.dumps(res, indent=2))
