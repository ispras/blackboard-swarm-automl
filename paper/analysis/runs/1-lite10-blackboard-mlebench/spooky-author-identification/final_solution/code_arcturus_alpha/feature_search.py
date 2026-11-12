import time, json, warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")

WS = "workspaces/arcturus_alpha"
train = pd.read_csv("data/train.csv")
y = np.load(f"{WS}/y.npy")
folds = np.load(f"{WS}/folds.npy")
NF = 5
COMMON = dict(sublinear_tf=True, strip_accents="unicode")


def _merge(base, kw):
    d = dict(base); d.update(kw); return d


def W(ng=(1, 2), min_df=2, lower=True, sw=None, **kw):
    base = dict(analyzer="word", token_pattern=r"\w{1,}", ngram_range=ng,
                min_df=min_df, lowercase=lower, stop_words=sw,
                sublinear_tf=True, strip_accents="unicode")
    return TfidfVectorizer(**_merge(base, kw))


def C(ng=(2, 5), min_df=2, analyzer="char_wb", lower=True, **kw):
    base = dict(analyzer=analyzer, ngram_range=ng, min_df=min_df,
                lowercase=lower, sublinear_tf=True, strip_accents="unicode")
    return TfidfVectorizer(**_merge(base, kw))


def run(name, vec_fns, clf_fn, save=None):
    oof = np.zeros((len(y), 3))
    t0 = time.time()
    for f in range(NF):
        tr, va = folds != f, folds == f
        trT, vaT = train["text"].values[tr], train["text"].values[va]
        Ptr, Pva = [], []
        for mk in vec_fns:
            v = mk()
            Ptr.append(v.fit_transform(trT)); Pva.append(v.transform(vaT))
        Xtr, Xva = hstack(Ptr).tocsr(), hstack(Pva).tocsr()
        clf = clf_fn()
        clf.fit(Xtr, y[tr])
        oof[va] = clf.predict_proba(Xva)
    ll = log_loss(y, oof, labels=[0, 1, 2])
    print(f"{name:46s} {Xtr.shape[1]:7d} feats  OOF={ll:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    if save:
        np.save(f"{WS}/oof_{save}.npy", oof)
    return ll, oof


LR10 = lambda: LogisticRegression(C=10, max_iter=3000, solver="lbfgs", n_jobs=4)
LR30 = lambda: LogisticRegression(C=30, max_iter=3000, solver="lbfgs", n_jobs=4)
LR100 = lambda: LogisticRegression(C=100, max_iter=3000, solver="lbfgs", n_jobs=4)

res = {}
# Feature configs with LR C=10
cfgs = {
    "w12md2_cwb25md2": [W, C],
    "w12md1_cwb25md1": [lambda: W((1, 2), min_df=1), lambda: C((2, 5), min_df=1)],
    "w13md1_cwb16md1": [lambda: W((1, 3), min_df=1), lambda: C((2, 6), min_df=1)],
    "w12md1_cwb25md1_c25md1": [lambda: W((1, 2), min_df=1), lambda: C((2, 5), min_df=1), lambda: C((2, 5), min_df=1, analyzer="char")],
    "w12md1_lower0": [lambda: W((1, 2), min_df=1, lower=False), lambda: C((2, 5), min_df=1, lower=False)],
    "w12md1_counts": [lambda: W((1, 2), min_df=1, use_idf=False), lambda: C((2, 5), min_df=1, use_idf=False)],
    "w12md1_nosub": [lambda: W((1, 2), min_df=1, sublinear_tf=False), lambda: C((2, 5), min_df=1, sublinear_tf=False)],
    "w12md1_sw": [lambda: W((1, 2), min_df=1, sw="english"), lambda: C((2, 5), min_df=1)],
}
for name, vf in cfgs.items():
    ll, _ = run(name, vf, LR10)
    res[name] = ll

print("\n-- C sweep on w12md1_cwb25md1 --")
vf = [lambda: W((1, 2), min_df=1), lambda: C((2, 5), min_df=1)]
for cn, cf in [("C3", LR10), ("", None)]:
    pass
for tag, cfn in [("C10", LR10), ("C30", LR30), ("C100", LR100)]:
    ll, _ = run(f"w12md1_cwb25md1_{tag}", vf, cfn, save=f"fs_{tag}")
    res[f"w12md1_cwb25md1_{tag}"] = ll

json.dump(res, open(f"{WS}/feature_search.json", "w"), indent=2)
print(json.dumps(res, indent=2))
