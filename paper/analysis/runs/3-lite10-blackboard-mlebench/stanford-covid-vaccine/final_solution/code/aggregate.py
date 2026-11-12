import os, re, json, glob
import numpy as np
from common import load_train, mcrmse, SCORE_IDX, WS

PD = os.path.join(WS, 'preds')
OUT = os.path.join(WS, 'agg')
os.makedirs(OUT, exist_ok=True)

tr = load_train()
n = len(tr['seq'])
snf = np.array([r['SN_filter'] for r in tr['rows']], dtype=bool)
mask = tr['mask'][:, :68]
yv = tr['y'][:, :68, :][:, :, SCORE_IDX]


def parse_prefix(fn):
    b = os.path.basename(fn)[:-4]
    m = re.match(r'^(.*)_s(\d+)_f(\d+)$', b)
    if m:
        return m.group(1)
    return b


files = sorted(glob.glob(os.path.join(PD, '*.npz')))
groups = {}
for f in files:
    groups.setdefault(parse_prefix(f), []).append(f)

summary = {}
for prefix, fs in groups.items():
    oof = np.zeros((n, 68, 3), dtype=np.float64)
    seen = np.zeros(n, dtype=bool)
    tests = []
    scores = []
    for f in fs:
        d = np.load(f)
        vi = d['val_idx']
        pv = d['oof'][:, :68, :][:, :, SCORE_IDX]
        oof[vi] = pv
        seen[vi] = True
        tests.append(d['test'][:, :68, :][:, :, SCORE_IDX])
        scores.append(float(d['score']))
    if not seen.all():
        continue
    test = np.mean(tests, axis=0)
    sc_all, per_all = mcrmse(yv, oof, mask, cols=[0,1,2])
    sc_sn, per_sn = mcrmse(yv[snf], oof[snf], mask[snf], cols=[0,1,2])
    sc_clip, per_clip = mcrmse(yv, np.maximum(oof, -0.5), mask, cols=[0,1,2])
    sc_sn_clip, per_sn_clip = mcrmse(yv[snf], np.maximum(oof[snf], -0.5), mask[snf], cols=[0,1,2])
    np.save(os.path.join(OUT, f'oof_{prefix}.npy'), oof.astype(np.float32))
    np.save(os.path.join(OUT, f'test_{prefix}.npy'), test.astype(np.float32))
    summary[prefix] = dict(n_runs=len(fs), all=sc_all, all_per=per_all,
                           sn=sc_sn, sn_per=per_sn,
                           all_clip=sc_clip, sn_clip=sc_sn_clip,
                           sn_per_clip=per_sn_clip, val_scores=scores)
    print(f"{prefix:18s} runs={len(fs):2d} all={sc_all:.4f} sn={sc_sn:.4f} "
          f"sn_clip={sc_sn_clip:.4f} per_sn={np.round(per_sn,3)}")

with open(os.path.join(OUT, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
