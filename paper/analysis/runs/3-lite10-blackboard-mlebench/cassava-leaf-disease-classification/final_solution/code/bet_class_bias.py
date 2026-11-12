import argparse, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = '/download_job_code/payload/sandbox'


def fit_bias(P, y, grid=None):
    if grid is None:
        grid = np.array([-1.2, -0.8, -0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.8, 1.2])
    logp = np.log(np.clip(P, 1e-9, None))
    b = np.zeros(P.shape[1])
    best = (logp + b).argmax(1)
    best_acc = (best == y).mean()
    for _ in range(4):
        improved = False
        for c in range(P.shape[1]):
            cur = b[c]
            for g in grid:
                b[c] = g
                a = ((logp + b).argmax(1) == y).mean()
                if a > best_acc + 1e-9:
                    best_acc, cur, improved = a, g, True
            b[c] = cur
        if not improved:
            break
    return b, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--cv', default=os.path.join(ROOT, 'workspaces/aldebaran_epsilon/cv_split.csv'))
    ap.add_argument('--sample', default=os.path.join(ROOT, 'data/sample_submission.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'workspaces/bellatrix_delta/bias_adj'))
    args = ap.parse_args()
    cv = pd.read_csv(args.cv); va = cv[cv.fold == 0].reset_index(drop=True)
    y = va.label.values
    P = np.clip(np.load(args.val).astype(np.float64), 1e-9, None); P /= P.sum(1, keepdims=True)
    T = np.clip(np.load(args.test).astype(np.float64), 1e-9, None); T /= T.sum(1, keepdims=True)
    base = (P.argmax(1) == y).mean()
    print(f'baseline fold0 acc={base:.5f}')

    skf = StratifiedKFold(5, shuffle=True, random_state=11)
    base_cv, adj_cv = [], []
    for tri, tei in skf.split(P, y):
        b, _ = fit_bias(P[tri], y[tri])
        base_cv.append((P[tei].argmax(1) == y[tei]).mean())
        adj_cv.append(((np.log(P[tei]) + b).argmax(1) == y[tei]).mean())
    print(f'nested-CV base={np.mean(base_cv):.5f}  bias-adj={np.mean(adj_cv):.5f}')
    b, full_acc = fit_bias(P, y)
    print('bias=', np.round(b, 3), 'in-sample acc=', round(full_acc, 5))
    delta = np.mean(adj_cv) - np.mean(base_cv)
    if delta > 0.001:
        print('APPLYING bias adjustment (CV positive)')
        labels = (np.log(T) + b).argmax(1)
    else:
        print('NOT applying (CV not positive enough)')
        labels = T.argmax(1)
    os.makedirs(args.out, exist_ok=True)
    sample = pd.read_csv(args.sample)
    pd.DataFrame({'image_id': sample.image_id, 'label': labels}).to_csv(os.path.join(args.out, 'submission.csv'), index=False)
    json.dump({'baseline': float(base), 'cv_base': float(np.mean(base_cv)), 'cv_adj': float(np.mean(adj_cv)),
               'bias': b.tolist(), 'applied': bool(delta > 0.001)}, open(os.path.join(args.out, 'meta.json'), 'w'), indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
