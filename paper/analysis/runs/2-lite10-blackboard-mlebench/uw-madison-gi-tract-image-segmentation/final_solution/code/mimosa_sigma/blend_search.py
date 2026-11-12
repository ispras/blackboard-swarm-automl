import os, sys, json, time, argparse, hashlib, itertools
import numpy as np
from scipy.spatial import cKDTree
import cv2
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/mimosa_sigma/code')
import simlib
from tune_postproc import postproc, decide
ROOT = '/download_job_code/payload/sandbox'
C = f'{ROOT}/workspaces/mimosa_sigma/cache'
CLASSES = simlib.CLASSES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', nargs='+', required=True)
    ap.add_argument('--thr', default='0.7,0.7,0.7')
    ap.add_argument('--cc2d', default='100,100,100')
    ap.add_argument('--cc3d', default='800,800,800')
    ap.add_argument('--step', type=float, default=0.25)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    t0 = time.time()
    study = json.load(open(f'{C}/train_study.json'))
    foldarr = np.load(f'{C}/train_fold.npy')
    va = np.where(foldarr == 0)[0]
    mask = np.load(f'{C}/train_mask.npy', mmap_mode='r')
    groups = {}
    for i in va:
        groups.setdefault(study[i], []).append(i)
    groups = {g: np.array(v) for g, v in groups.items()}
    gnames = sorted(groups)
    hv = lambda g: int(hashlib.md5(g.encode()).hexdigest(), 16) % 2
    tune = [g for g in gnames if hv(g) == 0]
    thrs = [float(x) for x in args.thr.split(',')]
    c2 = [int(x) for x in args.cc2d.split(',')]
    c3 = [int(x) for x in args.cc3d.split(',')]
    gt = {}; gtree = {}
    for g in gnames:
        idxs = groups[g]
        for c in range(3):
            v = np.ascontiguousarray(mask[idxs, c] > 0); gt[(g, c)] = v
            if v.any():
                gtree[(g, c)] = cKDTree(np.argwhere(v).astype(np.float32))
    probs = [np.load(p, mmap_mode='r') for p in args.probs]
    n = len(probs)
    step = args.step
    vals = np.round(np.arange(0, 1.0001, step), 3)
    cands = [w for w in itertools.product(vals, repeat=n) if 0.999 < sum(w) < 1.001]
    print(f'{len(cands)} weight combos, tune {len(tune)} groups', flush=True)
    scores = {w: [] for w in cands}
    for c in range(3):
        Ps = [np.asarray(p[:, c]) for p in probs]
        for w in cands:
            Pc = np.zeros((len(va), 320, 320), np.float32)
            for wi, Pi in zip(w, Ps):
                if wi > 0:
                    Pc += np.float32(wi) * Pi
            scs = []
            for g in tune:
                rows = np.searchsorted(va, groups[g])
                b = postproc(Pc[rows], thrs[c], c2[c], c3[c], 0)
                sc, d, h = decide(b, gt[(g, c)], gtree.get((g, c)))
                scs.append(sc)
            scores[w].append(float(np.mean(scs)))
        print('class', CLASSES[c], 'done', f'{time.time()-t0:.0f}s', flush=True)
        del Ps
    ranked = sorted(((np.mean(v), w, v) for w, v in scores.items()), reverse=True)
    for r in ranked[:8]:
        print(f'w={r[1]} mean_tune_combo={r[0]:.4f} per_class={[round(x,4) for x in r[2]]}', flush=True)
    if args.out:
        json.dump({'best': ranked[0][1], 'score': ranked[0][0], 'per_class': ranked[0][2],
                   'probs': args.probs, 'ranking': [(r[1], r[0], r[2]) for r in ranked[:8]]},
                  open(args.out, 'w'), indent=2)

if __name__ == '__main__':
    main()
