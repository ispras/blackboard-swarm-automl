"""Reproduce the final blended submission from member test predictions.

Reads `members.json` next to this file:
  {
    "root": "...",                      # optional sandbox root (default ../../..)
    "combiner": "weighted_sum",         # weighted sum of member predictions
    "members": [{"name": "...", "file": "preds/xxx_test.csv", "weight": 0.5}, ...],
    "postprocess": "postprocess_final.py"
  }
Each member CSV has columns `key,fare_amount` (or `key,pred`) in test.csv order.
The blend is then passed through the swarm's validated invalid-coordinate
postprocessor before writing `submission.csv`.
"""
import os, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def _find_root(start):
    d = start
    for _ in range(6):
        if os.path.exists(os.path.join(d, 'data', 'test.csv')):
            return d
        d = os.path.dirname(d)
    return os.path.abspath(os.path.join(start, '..', '..'))

ROOT = _find_root(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', default=os.path.join(HERE, 'members.json'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cfg = json.load(open(args.members))
    out = args.out or os.path.join(os.path.dirname(HERE), 'submission.csv')

    test = pd.read_csv(f'{ROOT}/data/test.csv', dtype={'key': str})
    keys = test['key'].values
    blend = np.zeros(len(test), dtype=np.float64)
    wsum = 0.0
    for m in cfg['members']:
        f = m['file']
        if not os.path.isabs(f):
            f = os.path.join(HERE, f)
        d = pd.read_csv(f, dtype={'key': str})
        col = 'pred' if 'pred' in d.columns else 'fare_amount'
        d = d.set_index('key')
        vals = d[col].reindex(keys).to_numpy(np.float64)
        assert not np.isnan(vals).any(), f'missing keys for {m["name"]}'
        blend += m['weight'] * vals
        wsum += m['weight']
    blend /= wsum
    print('combiner weight sum', wsum)

    pp_path = cfg.get('postprocess')
    if pp_path:
        pq = os.path.join(HERE, pp_path)
        import importlib.util
        spec = importlib.util.spec_from_file_location('pp', pq)
        pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
        blend = pp.apply_postprocess(blend, test)
    pd.DataFrame({'key': test['key'], 'fare_amount': blend}).to_csv(out, index=False)
    print('wrote', out, len(blend), 'min/mean/max', blend.min(), blend.mean(), blend.max())


if __name__ == '__main__':
    main()
