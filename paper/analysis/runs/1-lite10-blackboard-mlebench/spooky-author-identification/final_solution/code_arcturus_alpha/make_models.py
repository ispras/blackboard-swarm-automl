import sys, time, json, warnings, argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")

WS = "workspaces/arcturus_alpha"
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = np.load(f"{WS}/y.npy")
folds = np.load(f"{WS}/folds.npy")
NF = 5


def _v(base, kw):
    d = dict(base); d.update(kw); return TfidfVectorizer(**d)


def WV(ng=(1, 2), min_df=2, lower=True, **kw):
    return _v(dict(analyzer="word", token_pattern=r"\w{1,}", ngram_range=ng, min_df=min_df,
                   lowercase=lower, sublinear_tf=True, strip_accents="unicode"), kw)


def CV(ng=(2, 5), min_df=2, lower=True, analyzer="char_wb", **kw):
    return _v(dict(analyzer=analyzer, ngram_range=ng, min_df=min_df, lowercase=lower,
                   sublinear_tf=True, strip_accents="unicode"), kw)


FS = {
    "case": lambda: [WV((1, 2), 1, lower=False), CV((2, 5), 1, lower=False)],
    "case3": lambda: [WV((1, 2), 1, lower=False), CV((2, 5), 1, lower=False), CV((2, 5), 1, lower=False, analyzer="char")],
    "nocase": lambda: [WV((1, 2), 2), CV((2, 5), 2)],
    "case_w13": lambda: [WV((1, 3), 1, lower=False), CV((2, 6), 1, lower=False)],
}


def run(name, fsname, clf_fn):
    oof = np.zeros((len(y), 3))
    testp = np.zeros((len(test), 3))
    txt = train["text"].values
    ttxt = test["text"].values
    t0 = time.time()
    for f in range(NF):
        tr, va = folds != f, folds == f
        ptr, pva, pte = [], [], []
        for v in FS[fsname]():
            v.fit(txt[tr])
            ptr.append(v.transform(txt[tr]))
            pva.append(v.transform(txt[va]))
            pte.append(v.transform(ttxt))
        Xtr, Xva, Xte = hstack(ptr).tocsr(), hstack(pva).tocsr(), hstack(pte).tocsr()
        clf = clf_fn()
        clf.fit(Xtr, y[tr])
        oof[va] = clf.predict_proba(Xva)
        testp += clf.predict_proba(Xte) / NF
    ll = log_loss(y, oof, labels=[0, 1, 2])
    np.save(f"{WS}/oof_{name}.npy", oof)
    np.save(f"{WS}/test_{name}.npy", testp)
    print(f"{name:26s} OOF={ll:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return ll


MODELS = {
    "case_lr": ("case", lambda: LogisticRegression(C=10, max_iter=3000, solver="lbfgs", n_jobs=4)),
    "case_lrC5": ("case", lambda: LogisticRegression(C=5, max_iter=3000, solver="lbfgs", n_jobs=4)),
    "case3_lr": ("case3", lambda: LogisticRegression(C=10, max_iter=3000, solver="lbfgs", n_jobs=4)),
    "nocase_lr": ("nocase", lambda: LogisticRegression(C=10, max_iter=3000, solver="lbfgs", n_jobs=4)),
    "case_cnb": ("case", lambda: ComplementNB(alpha=0.3)),
    "nocase_cnb": ("nocase", lambda: ComplementNB(alpha=0.3)),
    "case_mnb": ("case", lambda: MultinomialNB(alpha=0.03)),
    "case_sgd": ("case", lambda: SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=100, tol=1e-4, random_state=42)),
    "case_svc": ("case", lambda: CalibratedClassifierCV(LinearSVC(C=0.5), method="sigmoid", cv=3)),
    "case_pa": ("case", lambda: CalibratedClassifierCV(PassiveAggressiveClassifier(C=0.1, max_iter=50, random_state=42), method="sigmoid", cv=3)),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    res = {}
    for name, (fs, mk) in MODELS.items():
        if args.only and name not in args.only:
            continue
        res[name] = run(name, fs, mk)
    print(json.dumps(res, indent=2))
