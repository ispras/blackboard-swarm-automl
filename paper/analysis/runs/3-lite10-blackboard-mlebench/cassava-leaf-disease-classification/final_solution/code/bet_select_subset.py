import argparse, os, itertools, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = '/download_job_code/payload/sandbox'


def norm(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, None); return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', nargs='+', required=True, help='name:val_probs[:val_ids]')
    ap.add_argument('--cv', default=os.path.join(ROOT, 'workspaces/aldebaran_epsilon/cv_split.csv'))
    args = ap.parse_args()
    cv = pd.read_csv(args.cv); va = cv[cv.fold == 0].reset_index(drop=True)
    y = va.label.values; va_ids = va.image_id.values
    names, V = [], []
    for spec in args.val:
        parts = spec.split(':')
        name, vp = parts[0], parts[1]; vi = parts[2] if len(parts) > 2 else None
        v = norm(np.load(vp))
        if vi and os.path.exists(vi):
            ids = pd.read_csv(vi).image_id.values
            if len(ids) == len(v) and not np.array_equal(ids, va_ids):
                order = pd.Index(ids); v = v[[order.get_loc(i) for i in va_ids]]
        names.append(name); V.append(v)
    V = np.stack(V); n = len(names)
    print('singles:', {names[i]: round(float((V[i].argmax(1) == y).mean()), 5) for i in range(n)})

    folds = list(StratifiedKFold(5, shuffle=True, random_state=7).split(V[0], y))
    results = []
    for r in range(1, n + 1):
        for sub in itertools.combinations(range(n), r):
            accs = []
            for tri, tei in folds:
                p = V[list(sub)][:, tei].mean(0)
                accs.append((p.argmax(1) == y[tei]).mean())
            accs = np.array(accs)
            ins = (V[list(sub)].mean(0).argmax(1) == y).mean()
            results.append((accs.mean(), accs.std() / np.sqrt(5), ins, sub))
    results.sort(key=lambda x: -x[0])
    print('\nTOP 12 subsets by nested-CV:')
    for m, se, ins, sub in results[:12]:
        print(f'  cv={m:.5f}±{se:.4f} in={ins:.5f}  {[names[i] for i in sub]}')
    best = results[0]
    # 1-SE rule: among subsets within 1 SE of best CV, pick the largest subset (more diversity)
    thr = best[0] - best[1]
    cand = [r for r in results if r[0] >= thr]
    cand.sort(key=lambda x: (-len(x[3]), -x[0]))
    chosen = cand[0]
    print(f'\n1-SE chosen: cv={chosen[0]:.5f} {[names[i] for i in chosen[3]]}')
    print('BEST_SUBSET=' + ','.join(names[i] for i in best[3]))
    print('CHOSEN_SUBSET=' + ','.join(names[i] for i in chosen[3]))


if __name__ == '__main__':
    main()
