"""Self-contained reproduction of final_solution/submission.csv.

Uses the saved base-model OOF/test probability matrices (base_oof.npz, base_test.npz)
and the exact meta-recipe from final_recipe.json.

Run from this directory:
    pyenv shell sis-black && python reproduce_submission.py
"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

EPS = 1e-15
def norm(p):
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(1, keepdims=True)

r = json.load(open('final_recipe.json'))
names = r['models']
W = r['w']; C = r['C']

train = pd.read_csv('../../data/train.csv')
y = train['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2}).values
te = pd.read_csv('../../data/test.csv')

O = np.stack([norm(np.load('base_oof.npz')[n]) for n in names])
T = np.stack([norm(np.load('base_test.npz')[n]) for n in names])
def feat(P):
    L = np.log(P)
    return np.concatenate([L, P], 2).transpose(1, 0, 2).reshape(len(P[0]), -1)
X, XT = feat(O), feat(T)

folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
lr = np.zeros((len(y), 3)); hg = np.zeros((len(y), 3)); xg = np.zeros((len(y), 3))
for i, j in folds:
    lr[j] = LogisticRegression(C=C, max_iter=8000).fit(X[i], y[i]).predict_proba(X[j])
    hg[j] = HistGradientBoostingClassifier(learning_rate=0.03, max_leaf_nodes=15, max_iter=200,
                                           l2_regularization=1.0, random_state=42).fit(X[i], y[i]).predict_proba(X[j])
    xg[j] = XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8,
                          colsample_bytree=0.6, reg_lambda=1.0, objective='multi:softprob',
                          tree_method='hist', n_jobs=2, random_state=42, verbosity=0).fit(X[i], y[i]).predict_proba(X[j])
oof = norm(np.exp(W[0]*np.log(norm(lr)) + W[1]*np.log(norm(hg)) + W[2]*np.log(norm(xg))))
print('OOF logloss:', log_loss(y, oof))

lr_t = norm(LogisticRegression(C=C, max_iter=8000).fit(X, y).predict_proba(XT))
hg_t = norm(HistGradientBoostingClassifier(learning_rate=0.03, max_leaf_nodes=15, max_iter=200,
                                           l2_regularization=1.0, random_state=42).fit(X, y).predict_proba(XT))
xg_t = norm(XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8,
                          colsample_bytree=0.6, reg_lambda=1.0, objective='multi:softprob',
                          tree_method='hist', n_jobs=2, random_state=42, verbosity=0).fit(X, y).predict_proba(XT))
test = norm(np.exp(W[0]*np.log(lr_t) + W[1]*np.log(hg_t) + W[2]*np.log(xg_t)))
test = np.clip(test, 1e-9, 1); test = test / test.sum(1, keepdims=True)
sub = pd.DataFrame({'id': te['id'], 'EAP': test[:, 0], 'HPL': test[:, 1], 'MWS': test[:, 2]})
sub.to_csv('submission_reproduced.csv', index=False)
print('wrote submission_reproduced.csv', sub.shape)
