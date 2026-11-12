import time, json, warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")

WS = "workspaces/arcturus_alpha"
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = np.load(f"{WS}/y.npy")
folds = np.load(f"{WS}/folds.npy")
NF = 5


def case3(trT, otherT):
    vs = [
        TfidfVectorizer(analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1, 2), min_df=1,
                        lowercase=False, sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                        lowercase=False, sublinear_tf=True, strip_accents="unicode"),
        TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1,
                        lowercase=False, sublinear_tf=True, strip_accents="unicode"),
    ]
    tr = [v.fit_transform(trT) for v in vs]
    oth = [v.transform(otherT) for v in vs]
    return hstack(tr).tocsr(), hstack(oth).tocsr()


def run(name, make, svd_dim=300):
    oof = np.zeros((len(y), 3)); testp = np.zeros((len(test), 3))
    txt = train["text"].values; ttxt = test["text"].values
    t0 = time.time()
    for f in range(NF):
        tr, va = folds != f, folds == f
        Xtr, Xva = case3(txt[tr], txt[va])
        _, Xte = case3(txt[tr], ttxt)
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        Ztr = svd.fit_transform(Xtr); Zva = svd.transform(Xva); Zte = svd.transform(Xte)
        sc = StandardScaler()
        Ztr = sc.fit_transform(Ztr); Zva = sc.transform(Zva); Zte = sc.transform(Zte)
        clf = make()
        clf.fit(Ztr, y[tr])
        oof[va] = clf.predict_proba(Zva)
        testp += clf.predict_proba(Zte) / NF
    ll = log_loss(y, oof, labels=[0, 1, 2])
    np.save(f"{WS}/oof_{name}.npy", oof); np.save(f"{WS}/test_{name}.npy", testp)
    print(f"{name:20s} OOF={ll:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return ll


def mlp():
    return MLPClassifier(hidden_layer_sizes=(256, 128), alpha=3e-3, batch_size=128,
                         learning_rate_init=1e-3, max_iter=120, early_stopping=False,
                         random_state=42)


def nyst():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(Nystroem(gamma=0.02, n_components=1200, random_state=42),
                         LogisticRegression(C=10, max_iter=2000, n_jobs=8))


if __name__ == "__main__":
    res = {}
    res["mlp_svd"] = run("mlp_svd", mlp, 300)
    res["nystroem_rbf"] = run("nystroem_rbf", nyst, 200)
    print(json.dumps(res, indent=2))
