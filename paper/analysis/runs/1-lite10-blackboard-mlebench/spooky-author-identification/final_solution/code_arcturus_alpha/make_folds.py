import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("data/train.csv")
y = train["author"].map({"EAP": 0, "HPL": 1, "MWS": 2}).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.zeros(len(y), dtype=int)
for f, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
    folds[va] = f
np.save("workspaces/arcturus_alpha/folds.npy", folds)
np.save("workspaces/arcturus_alpha/y.npy", y)
print("class counts", np.bincount(y))
for f in range(5):
    print(f, "fold size", (folds == f).sum(), "dist", np.bincount(y[folds == f], minlength=3))
print("saved folds.npy, y.npy")
