import os, sys, json, time, argparse
import numpy as np, pandas as pd, cv2
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/mimosa_sigma/code')
import simlib
from tune_postproc import postproc
ROOT = '/download_job_code/payload/sandbox'
C = f'{ROOT}/workspaces/mimosa_sigma/cache'
CLASSES = simlib.CLASSES
cv2.setNumThreads(4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', nargs='+', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=None)
    ap.add_argument('--recipe', default='')
    ap.add_argument('--thr', default='0.60,0.60,0.60')
    ap.add_argument('--cc2d', default='200,200,200')
    ap.add_argument('--cc3d', default='0,0,0')
    ap.add_argument('--keeptop', default='0,0,0')
    ap.add_argument('--out', default=f'{ROOT}/final_solution/submission.csv')
    args = ap.parse_args()
    t0 = time.time()
    ids = json.load(open(f'{C}/test_ids.json'))
    study = json.load(open(f'{C}/test_study.json'))
    hw = np.load(f'{C}/test_hw.npy')
    n = len(ids)
    ws = np.array(args.weights or [1.0] * len(args.probs), np.float32); ws /= ws.sum()
    if args.recipe:
        rec = json.load(open(args.recipe))
        thr = [rec['classes'][c]['thr'] for c in CLASSES]
        cc2 = [rec['classes'][c]['cc2d'] for c in CLASSES]
        cc3 = [rec['classes'][c]['cc3d'] for c in CLASSES]
        kt = [rec['classes'][c]['keep_top'] for c in CLASSES]
    else:
        thr = [float(x) for x in args.thr.split(',')]
        cc2 = [int(x) for x in args.cc2d.split(',')]
        cc3 = [int(x) for x in args.cc3d.split(',')]
        kt = [int(x) for x in args.keeptop.split(',')]
    # blend per class to bound memory
    blend = np.zeros((n, 3, 320, 320), np.float16)
    for k, p in enumerate(args.probs):
        a = np.load(p, mmap_mode='r')
        assert len(a) == n, (p, len(a), n)
        s = a.shape[-1]
        if s == 320:
            src = a
        else:
            src = None
        for i in range(n):
            for c in range(3):
                v = np.asarray(a[i, c], dtype=np.float32)
                if s != 320:
                    v = cv2.resize(v, (320, 320), interpolation=cv2.INTER_LINEAR)
                blend[i, c] += np.float16(ws[k] * v)
        print('loaded', p, f'{time.time()-t0:.0f}s', flush=True)
    groups = {}
    for i, g in enumerate(study):
        groups.setdefault(g, []).append(i)
    rows = []
    for g, idxs in groups.items():
        idxs = np.array(idxs)
        for c in range(3):
            vol = np.asarray(blend[idxs, c], dtype=np.float32)
            b = postproc(vol, thr[c], cc2[c], cc3[c], kt[c])
            for z, i in enumerate(idxs):
                m = cv2.resize(b[z].astype(np.uint8), (int(hw[i, 1]), int(hw[i, 0])),
                               interpolation=cv2.INTER_NEAREST)
                rows.append((ids[i], CLASSES[c], simlib.rle_encode(m)))
    df = pd.DataFrame(rows, columns=['id', 'class', 'predicted'])
    sub = pd.read_csv(f'{ROOT}/data/test.csv')
    out = sub[['id', 'class']].merge(df, on=['id', 'class'], how='left')
    out['predicted'] = out['predicted'].fillna('')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print('wrote', args.out, len(out), 'rows;', int((out.predicted != '').sum()), 'nonempty', f'{time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
