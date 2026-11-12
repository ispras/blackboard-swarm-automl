import time, json, warnings, sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")

WS = "workspaces/arcturus_alpha"
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = np.load(f"{WS}/y.npy")
folds = np.load(f"{WS}/folds.npy")
NF = 5


def vs_case():
    return [TfidfVectorizer(analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1, 2), min_df=1,
                            lowercase=False, sublinear_tf=True, strip_accents="unicode"),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                            lowercase=False, sublinear_tf=True, strip_accents="unicode"),
            TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1,
                            lowercase=False, sublinear_tf=True, strip_accents="unicode")]


def mat(trT, otherT):
    vs = vs_case()
    tr = [v.fit_transform(trT) for v in vs]
    oth = [v.transform(otherT) for v in vs]
    return hstack(tr).tocsr(), hstack(oth).tocsr()


def teacher_models():
    return [LogisticRegression(C=10, max_iter=3000, solver="lbfgs", n_jobs=8),
            ComplementNB(alpha=0.3), MultinomialNB(alpha=0.03)]


def run(mode="hard", conf_pow=1.0, thresh=0.0):
    oof = np.zeros((len(y), 3)); testp = np.zeros((len(test), 3))
    txt = train["text"].values; ttxt = test["text"].values
    t0 = time.time()
    for f in range(NF):
        tr, va = folds != f, folds == f
        Xtr, Xva = mat(txt[tr], txt[va])
        _, Xte = mat(txt[tr], ttxt)
        # teacher ensemble on train-fold
        soft = 0.0
        for m in teacher_models():
            m.fit(Xtr, y[tr])
            soft = soft + m.predict_proba(Xte)
        soft /= len(teacher_models())
        hard = soft.argmax(1)
        conf = soft.max(1)
        keep = conf >= thresh
        Xs = vstack([Xtr, Xte[keep]]).tocsr()
        ys = np.concatenate([y[tr], hard[keep]])
        ws = np.concatenate([np.ones(tr.sum()), conf[keep] ** conf_pow])
        stu = LogisticRegression(C=10, max_iter=3000, solver="lbfgs", n_jobs=8)
        stu.fit(Xs, ys, sample_weight=ws)
        oof[va] = stu.predict_proba(Xva)
        testp += stu.predict_proba(Xte) / NF
    ll = log_loss(y, oof, labels=[0, 1, 2])
    print(f"pseudo mode={mode} pow={conf_pow} thr={thresh} OOF={ll:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return ll, oof, testp


if __name__ == "__main__":
    res = {}
    base = 0.36707
    best = None
    for cp, th in [(0.0, 0.0), (1.0, 0.0), (1.0, 0.6), (2.0, 0.6), (1.0, 0.8)]:
        ll, oof, testp = run("hard", cp, th)
        res[f"pow{cp}_thr{th}"] = ll
        if best is None or ll < best[0]:
            best = (ll, oof, testp, cp, th)
    json.dump(res, open(f"{WS}/pseudo_results.json", "w"), indent=2)
    print("BASE case3_lr =", base, "->", json.dumps(res, indent=2))
    if best[0] < base:
        np.save(f"{WS}/oof_pseudo_lr.npy", best[1]); np.save(f"{WS}/test_pseudo_lr.npy", best[2])
        print("SAVED best pseudo", best[0], best[3], best[4])
