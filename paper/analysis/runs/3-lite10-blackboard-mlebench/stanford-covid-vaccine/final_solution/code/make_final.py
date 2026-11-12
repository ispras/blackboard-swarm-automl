"""Assemble the swarm's best agreed solution into final_solution/.
Loads the latest greedy blend produced by ensemble2.py (ens/test_ens_greedy.npy),
falls back to recomputing greedy if missing.
Usage: python make_final.py --write
"""
import os, glob, json, shutil, sys, numpy as np
from common import load_train, load_test, mcrmse, SCORE_IDX, WS, write_submission

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL = os.path.join(ROOT, 'final_solution')
AGG = os.path.join(WS, 'agg'); ENS = os.path.join(WS, 'ens')
tr = load_train(); te = load_test()
n = len(tr['seq'])
snf = np.array([r['SN_filter'] for r in tr['rows']], dtype=bool)
mask = tr['mask'][:, :68]
yv = tr['y'][:, :68, :][:, :, SCORE_IDX]

tf = os.path.join(ENS, 'test_ens_greedy.npy')
wf = os.path.join(ENS, 'greedy_weights.json')
if os.path.exists(tf) and os.path.exists(wf):
    test_pred = np.maximum(np.load(tf).astype(np.float64), -0.5)
    meta = json.load(open(wf))
    names, W = meta['models'], np.array(meta['W'])
    s_sn = meta.get('nested_sn')
    s_all = meta.get('nested_all')
    print('loaded blend from ens/ (nested sn=%s all=%s)' % (s_sn, s_all))
else:
    raise SystemExit('missing ens/test_ens_greedy.npy - run ensemble2.py first')

# full 5-col / 107-pos base from strong full 5-col swarm predictions
posmean = np.zeros((107, 5))
for c in range(5):
    for p in range(107):
        v = tr['y'][:, p, c][tr['mask'][:, p] > 0]
        posmean[p, c] = v.mean() if len(v) else 0.0
full = np.tile(posmean[None, :, :], (len(te['seq']), 1, 1))
cands = [
    os.path.join(ROOT, 'workspaces/fomalhaut_zeta/oof/test5_lstm_erraw.npy'),
    os.path.join(ROOT, 'workspaces/fomalhaut_zeta/oof/test5_lstm_werr6.npy'),
    os.path.join(ROOT, 'workspaces/fomalhaut_zeta/oof/test5_lstm_fwd6.npy'),
    os.path.join(ROOT, 'workspaces/deneb_upsilon/run_d_tta/test.npy'),
    os.path.join(ROOT, 'workspaces/deneb_upsilon/run_seed1_tta/test.npy'),
    os.path.join(ROOT, 'workspaces/deneb_upsilon/run_seed2_tta/test.npy'),
]
arrs = [np.load(c) for c in cands if os.path.exists(c)]
arrs = [a for a in arrs if a.shape == (len(te['seq']), 107, 5)]
if arrs:
    full = np.mean(arrs, axis=0).astype(np.float64)
    print('full5 base from', len(arrs), 'files')
for k, c in enumerate(SCORE_IDX):
    full[:, :68, c] = test_pred[:, :, k]
full = np.maximum(full, -0.5)  # test filter: all targets > -0.5

print('weights:', {names[i]: round(float(W[i]), 3) for i in range(len(names)) if W[i] > 1e-4})
print('test_blend mean', np.round(test_pred.mean((0, 1)), 4))

if '--write' in sys.argv:
    os.makedirs(FINAL, exist_ok=True)
    write_submission(full, te['ids'], os.path.join(FINAL, 'submission.csv'))
    json.dump(dict(models=names, weights=W.tolist(), nested_all=s_all, nested_sn=s_sn),
              open(os.path.join(FINAL, 'blend_weights.json'), 'w'), indent=2)
    np.save(os.path.join(FINAL, 'test_blend.npy'), test_pred.astype(np.float32))
    np.save(os.path.join(FINAL, 'oof_blend.npy'), np.load(os.path.join(ENS, 'oof_ens_greedy.npy')).astype(np.float32))
    code = os.path.join(FINAL, 'code'); os.makedirs(code, exist_ok=True)
    for f in ['common.py', 'model.py', 'train.py', 'aggregate.py', 'ensemble2.py', 'make_final.py']:
        shutil.copy(os.path.join(WS, f), os.path.join(code, f))
    open(os.path.join(code, 'run_all.sh'), 'w').write(
        '#!/bin/bash\n# Reproduce base models (see run_cv.sh / run_seed.sh / run_v2.sh), then:\n'
        'python aggregate.py && python ensemble2.py && python make_final.py --write\n')
    selected = {names[i]: round(float(W[i]), 3) for i in range(len(names)) if W[i] > 1e-4}
    readme = f'''# OpenVaccine mRNA degradation - swarm solution

Metric: **MCRMSE** on `reactivity, deg_Mg_pH10, deg_Mg_50C` (first 68 positions).
Greedy (Caruana) ensemble selection over out-of-fold predictions of many independently
trained sequence models (canonical KFold(5, shuffle, seed=42), a few with alternate split
seeds). Predictions clipped at -0.5 (test filter guarantees all targets > -0.5).

Nested-CV OOF MCRMSE (honest): **{s_sn:.4f}** (SN_filter==1 validation subset; all-rows {s_all:.4f}).

## Selected models and greedy weights
{json.dumps(selected, indent=2)}

Provenance: `fom_*` = fomalhaut_zeta (conv+BiLSTM variants: forward-only seed-averaged,
inverse-variance / error-weighted); `deneb_*` = deneb_upsilon (CNN+BiGRU with rev-comp
augmentation+TTA, transformer, Huber, error-weighted variants); `errw*`, `cnn*/lstm*/gru*/v2*`
= achernar_alpha (CNN / BiLSTM / BiGRU, including inverse-measurement-variance loss weighting
using the provided `*_error_*` columns). All models trained only on `data/train.json`.

## Files
- `submission.csv` - final predictions (107 positions x 5 columns; only the 3 scored columns
  at the first 68 positions are scored).
- `code/` - training (common.py, model.py, train.py), aggregation, ensemble selection, assembly.
- `blend_weights.json`, `oof_blend.npy`, `test_blend.npy`.

No external pretrained model checkpoints were used for this solution.
'''
    open(os.path.join(FINAL, 'README.md'), 'w').write(readme)
    print('wrote final_solution/submission.csv + README + code/')
else:
    print('dry run; pass --write')
