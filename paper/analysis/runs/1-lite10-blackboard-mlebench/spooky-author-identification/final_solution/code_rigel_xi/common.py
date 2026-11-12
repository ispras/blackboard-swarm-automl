import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA = "/download_job_code/payload/sandbox/data"
WS = "/download_job_code/payload/sandbox/workspaces/rigel_xi"
FINAL = "/download_job_code/payload/sandbox/final_solution"
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 42
NFOLD = 5


def load_data():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = tr["author"].map({c: i for i, c in enumerate(CLASSES)}).values
    return tr, te, y


def folds(y, n=NFOLD, seed=SEED):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def logloss(y, p):
    from sklearn.metrics import log_loss
    return log_loss(y, p, labels=[0, 1, 2])
