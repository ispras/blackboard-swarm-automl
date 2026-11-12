import os, sys, time, json
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline

WS = "workspaces/arcturus_alpha"
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = np.load(f"{WS}/y.npy")
folds = np.load(f"{WS}/folds.npy")
n_folds = 5
CLASSES = ["EAP", "HPL", "MWS"]


def build_word(train_text, test_text, test_only):
    v = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, min_df=2,
        strip_accents="unicode", analyzer="word", token_pattern=r"\w{1,}",
    )
    Xtr = v.fit_transform(train_text)
    Xte = v.transform(test_text)
    return Xtr, Xte


def build_char(train_text, test_text):
    v = TfidfVectorizer(
        ngram_range=(2, 5), sublinear_tf=True, min_df=2,
        strip_accents="unicode", analyzer="char_wb",
    )
    Xtr = v.fit_transform(train_text)
    Xte = v.transform(test_text)
    return Xtr, Xte


def cv_model(name, make_clf, feats=("word", "char")):
    oof = np.zeros((len(y), 3))
    test_pred = np.zeros((len(test), 3))
    t0 = time.time()
    for f in range(n_folds):
        tr = folds != f
        va = folds == f
        tr_txt = train["text"].values[tr]
        va_txt = train["text"].values[va]
        parts_tr, parts_va = [], []
        if "word" in feats:
            w_tr, w_va = build_word(tr_txt, va_txt, False)
            parts_tr.append(w_tr); parts_va.append(w_va)
        if "char" in feats:
            c_tr, c_va = build_char(tr_txt, va_txt)
            parts_tr.append(c_tr); parts_va.append(c_va)
        Xtr = hstack(parts_tr).tocsr()
        Xva = hstack(parts_va).tocsr()
        clf = make_clf()
        clf.fit(Xtr, y[tr])
        oof[va] = clf.predict_proba(Xva)
    ll = log_loss(y, oof, labels=[0, 1, 2])
    print(f"{name:28s} OOF logloss = {ll:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return oof, ll


if __name__ == "__main__":
    results = {}
    cfgs = {
        "lr_word_char_C3": lambda: LogisticRegression(C=3.0, max_iter=2000, solver="lbfgs", multi_class="multinomial", n_jobs=4),
        "lr_word_char_C10": lambda: LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs", multi_class="multinomial", n_jobs=4),
        "lr_word_only": lambda: LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs", multi_class="multinomial", n_jobs=4),
        "lr_char_only": lambda: LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs", multi_class="multinomial", n_jobs=4),
        "mnb_word_char": lambda: MultinomialNB(alpha=0.03),
        "cnb_word_char": lambda: ComplementNB(alpha=0.3),
        "sgd_word_char": lambda: SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=50, tol=1e-4, random_state=42),
    }
    for name, mk in cfgs.items():
        feats = ("word",) if name == "lr_word_only" else ("char",) if name == "lr_char_only" else ("word", "char")
        oof, ll = cv_model(name, mk, feats)
        np.save(f"{WS}/oof_baseline_{name}.npy", oof)
        results[name] = ll
    with open(f"{WS}/baseline_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
