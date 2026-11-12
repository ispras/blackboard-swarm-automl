import os, sys, json, argparse
import numpy as np
ROOT = '/download_job_code/payload/sandbox'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', nargs='+', required=True)
    ap.add_argument('--weights', default='')            # w1,w2,... same for all classes
    ap.add_argument('--weights-per-class', default='')  # "w1,w2;w1,w2;w1,w2"
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=512)
    args = ap.parse_args()
    ps = [np.load(p, mmap_mode='r') for p in args.probs]
    n = len(ps[0])
    for p in ps:
        assert len(p) == n, (p, len(p), n)
    if args.weights_per_class:
        W = [np.array([float(x) for x in g.split(',')]) for g in args.weights_per_class.split(';')]
        assert len(W) == 3
        for w in W:
            assert len(w) == len(ps); w /= w.sum()
    else:
        w = np.array([float(x) for x in args.weights.split(',')]); w /= w.sum()
        W = [w, w, w]
    print('per-class weights', [list(np.round(x, 3)) for x in W], flush=True)
    out = np.lib.format.open_memmap(args.out, mode='w+', dtype=np.float16, shape=(n, 3, 320, 320))
    for s in range(0, n, args.chunk):
        e = min(n, s + args.chunk)
        acc = np.zeros((e - s, 3, 320, 320), np.float32)
        for c in range(3):
            for k, p in enumerate(ps):
                if W[c][k] > 0:
                    acc[:, c] += np.float32(W[c][k]) * np.asarray(p[s:e, c], dtype=np.float32)
        out[s:e] = acc.astype(np.float16)
        if s % (args.chunk * 5) == 0:
            print('row', s, flush=True)
    out.flush()
    print('wrote', args.out, f'{os.path.getsize(args.out)/1e9:.2f}GB')

if __name__ == '__main__':
    main()
