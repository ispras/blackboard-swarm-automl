import sys, os, argparse
import numpy as np, pandas as pd

ROOT = '/download_job_code/payload/sandbox'
CACHE = f'{ROOT}/workspaces/regulus_alpha/cache'
CLASSES = ['large_bowel', 'small_bowel', 'stomach']


def origin_hw(path):
    b = os.path.basename(path).replace('.png', '').split('_')
    return int(b[3]), int(b[2])


def check_rle(s, HW):
    if not isinstance(s, str) or len(s.strip()) == 0:
        return True, 'empty'
    v = s.split()
    if len(v) % 2 != 0:
        return False, 'odd tokens'
    a = np.array(v, dtype=np.int64)
    starts, lens = a[0::2], a[1::2]
    if (starts < 1).any():
        return False, 'start<1'
    if (lens < 1).any():
        return False, 'len<1'
    if (np.diff(starts) <= 0).any():
        return False, 'starts not strictly increasing'
    ends = starts + lens - 1
    if (starts[1:] <= ends[:-1]).any():
        return False, 'overlap/duplicate'
    if ends[-1] > HW:
        return False, f'end>HW {ends[-1]}>{HW}'
    return True, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sub', default=f'{ROOT}/final_solution/submission.csv')
    a = ap.parse_args()
    sub = pd.read_csv(a.sub, keep_default_na=False)
    test = pd.read_csv(f'{ROOT}/data/test.csv')
    meta = pd.read_csv(f'{CACHE}/test_meta_320.csv').set_index('id')
    print('rows', len(sub), 'cols', list(sub.columns))
    ok_order = (len(sub) == len(test) and (sub['id'].values == test['id'].values).all()
                and (sub['class'].values == test['class'].values).all())
    print('order/id/class match test.csv:', ok_order)
    bad = []
    for i, (sid, cls, pred) in enumerate(zip(sub['id'], sub['class'], sub['predicted'])):
        h, w = origin_hw(meta.loc[sid, 'path'])
        ok, msg = check_rle(pred, h * w)
        if not ok:
            bad.append((sid, cls, msg))
            if len(bad) >= 10:
                break
    print('RLE invalid rows (first 10):', bad if bad else 'NONE')
    for cls in CLASSES:
        d = sub[sub['class'] == cls]
        ne = (d['predicted'].astype(str).str.len() > 0).mean()
        print(f'{cls}: nonempty frac {ne:.4f}')
    # predicted/GT area sanity at 320-decode on a sample
    print('OK' if ok_order and not bad else 'PROBLEM')


if __name__ == '__main__':
    main()
