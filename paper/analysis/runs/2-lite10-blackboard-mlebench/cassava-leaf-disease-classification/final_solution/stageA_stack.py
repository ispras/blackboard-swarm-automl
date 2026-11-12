import os, argparse
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
P = [f'p{i}' for i in range(5)]


def norm(a):
    return a / np.clip(a.sum(-1, keepdims=True), 1e-9, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='append', required=True,
                    help='name:val_csv:test_csv')
    ap.add_argument('--C', type=float, default=0.3)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default='stageA_stack_out')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    specs = [s.split(':', 2) for s in a.full]
    vdf = [pd.read_csv(s[1]).set_index('image_id') for s in specs]
    tdf = [pd.read_csv(s[2]).set_index('image_id') for s in specs]

    ids = vdf[0].index
    for d in vdf[1:]:
        ids = ids.intersection(d.index)
    ids = pd.Index(sorted(ids))
    y = vdf[0].loc[ids, 'label'].values.astype(int)
    X = np.concatenate([norm(d.loc[ids, P].values) for d in vdf], 1)

    tids = tdf[0].index
    for d in tdf[1:]:
        tids = tids.intersection(d.index)
    tids = pd.Index(sorted(tids))
    Xt = np.concatenate([norm(d.loc[tids, P].values) for d in tdf], 1)
    print('train', X.shape, 'test', Xt.shape)

    oof = np.zeros((len(y), 5))
    cnt = np.zeros(len(y))
    test_p = np.zeros((len(tids), 5))
    ntest = 0
    for seed in range(a.seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 100 + 7)
        for tr, va in skf.split(X, y):
            lr = LogisticRegression(C=a.C, max_iter=3000)
            lr.fit(X[tr], y[tr])
            oof[va] += lr.predict_proba(X[va])
            cnt[va] += 1
            test_p += lr.predict_proba(Xt)
            ntest += 1
    oof = norm(oof / np.clip(cnt, 1, None)[:, None])
    test_p = norm(test_p / ntest)
    print('stacker OOF acc', (oof.argmax(1) == y).mean())
    o = pd.DataFrame(oof, columns=P)
    o.insert(0, 'image_id', ids.values); o['label'] = y
    o.to_csv(os.path.join(a.out, 'E1_oof.csv'), index=False)
    t = pd.DataFrame(test_p, columns=P)
    t.insert(0, 'image_id', tids.values)
    t.to_csv(os.path.join(a.out, 'E1_test.csv'), index=False)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
