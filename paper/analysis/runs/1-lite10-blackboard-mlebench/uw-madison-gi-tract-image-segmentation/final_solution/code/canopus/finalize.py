import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuse_probs import fuse
from metric import hd_score, points_of, CLASSES
from postproc3d import clean_volume

ROOT = "/download_job_code/payload/sandbox"
CACHE_TR = f"{ROOT}/data/cache_train_320"
CACHE_TE = f"{ROOT}/data/cache_test_320"

def canon_ids():
    idx = pd.read_csv(f'{CACHE_TR}/index.csv')
    vc = set(json.load(open(f'{ROOT}/workspaces/regulus_alpha/val_cases.json'))['val_cases'])
    val = idx[idx['case'].isin(vc)]['id'].values
    test = pd.read_csv(f'{CACHE_TE}/index.csv')['id'].values
    pd.DataFrame({'id': val}).to_csv(f'{ROOT}/workspaces/canopus_phi/canon_val_ids.csv', index=False)
    pd.DataFrame({'id': test}).to_csv(f'{ROOT}/workspaces/canopus_phi/canon_test_ids.csv', index=False)
    return val, test

def class_eval(P, gt, groups, c, thr, min_vol=0, msa=0):
    pv = P[:, c] > int(thr * 255)
    g = (gt == c + 1)
    inter = np.logical_and(pv, g).reshape(len(pv), -1).sum(1)
    s = pv.reshape(len(pv), -1).sum(1) + g.reshape(len(g), -1).sum(1)
    dice = float(np.where(s > 0, 2 * inter / np.maximum(s, 1), 1.0).mean())
    hds = []
    for _, ii in groups:
        p = pv[ii]; gg = (gt[ii] == c + 1)
        if min_vol or msa:
            p = clean_volume(p, min_vol=min_vol, min_slice_area=msa)
        hds.append(hd_score(points_of(p), points_of(gg)))
    h = float(np.mean(hds))
    return dice, h, 0.4 * dice + 0.6 * h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val_probs', nargs='+', required=True)
    ap.add_argument('--val_ids', nargs='+', required=True)
    ap.add_argument('--test_probs', nargs='+', required=True)
    ap.add_argument('--test_ids', nargs='+', required=True)
    ap.add_argument('--weights', nargs='+', type=float, default=None)
    ap.add_argument('--thrs', default='0.25,0.30,0.35,0.40,0.45,0.50')
    ap.add_argument('--postproc', action='store_true')
    ap.add_argument('--out_submission', default=f'{ROOT}/final_solution/submission.csv')
    ap.add_argument('--tag', default='fused')
    a = ap.parse_args()
    assert len(a.val_probs) == len(a.val_ids) == len(a.test_probs) == len(a.test_ids)
    W = a.weights or [1.0] * len(a.val_probs)
    val, test = canon_ids()
    WS = f'{ROOT}/workspaces/canopus_phi'
    val_f = f'{WS}/{a.tag}_val_probs.npy'; test_f = f'{WS}/{a.tag}_test_probs.npy'
    print('fusing val...', flush=True)
    fuse(list(zip(a.val_probs, a.val_ids)), W, val, val_f)
    print('fusing test...', flush=True)
    fuse(list(zip(a.test_probs, a.test_ids)), W, test, test_f)

    index = pd.read_csv(f'{CACHE_TR}/index.csv'); mp = {v: i for i, v in enumerate(index['id'])}
    rows = np.array([mp[v] for v in val]); vi = index.iloc[rows].reset_index(drop=True)
    gt = np.asarray(np.load(f'{CACHE_TR}/masks_u8.npy', mmap_mode='r')[rows])
    P = np.asarray(np.load(val_f, mmap_mode='r'))
    groups = [(cd, g.index.values) for cd, g in vi.groupby('case_day')]
    thrs = [float(x) for x in a.thrs.split(',')]
    cfg = {}
    for c in range(3):
        best = None
        for t in thrs:
            d, h, combo = class_eval(P, gt, groups, c, t)
            if best is None or combo > best['combo']:
                best = dict(thr=t, min_vol=0, msa=0, dice=d, hd=h, combo=combo)
        if a.postproc:
            for mv in [0, 50, 150, 400]:
                for msa in [0, 30]:
                    d, h, combo = class_eval(P, gt, groups, c, best['thr'], mv, msa)
                    if combo > best['combo']:
                        best = dict(thr=best['thr'], min_vol=mv, msa=msa, dice=d, hd=h, combo=combo)
        cfg[c] = best
        print('class', CLASSES[c], json.dumps(best), flush=True)
    overall = dict(dice=float(np.mean([cfg[c]['dice'] for c in range(3)])),
                   hd=float(np.mean([cfg[c]['hd'] for c in range(3)])),
                   combo=float(np.mean([cfg[c]['combo'] for c in range(3)])))
    print('OVERALL', json.dumps(overall), flush=True)
    out_cfg = dict(models=a.val_probs, weights=W, per_class=cfg, overall=overall)
    json.dump(out_cfg, open(f'{WS}/{a.tag}_config.json', 'w'), indent=1)

    # make submission with per-class thresholds/postproc, C order
    from make_submission import rle_encode
    import cv2
    index_te = pd.read_csv(f'{CACHE_TE}/index.csv')
    Pf = np.asarray(np.load(test_f, mmap_mode='r'))
    pred = np.zeros((len(index_te), 3, 320, 320), bool)
    for c in range(3):
        pred[:, c] = Pf[:, c] > int(cfg[c]['thr'] * 255)
    for cd, g in index_te.groupby('case_day'):
        ii = g.index.values
        for c in range(3):
            if cfg[c]['min_vol'] or cfg[c]['msa']:
                pred[ii, c] = clean_volume(pred[ii, c], cfg[c]['min_vol'], 0, cfg[c]['msa'])
    sub_rows = []
    for i in range(len(index_te)):
        r = index_te.iloc[i]; oh, ow = int(r.orig_h), int(r.orig_w)
        for c, cls in enumerate(CLASSES):
            m = pred[i, c]
            if m.sum() == 0:
                sub_rows.append((r.id, cls, '')); continue
            mr = cv2.resize(m.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
            sub_rows.append((r.id, cls, rle_encode(mr, 'C')))
        if i % 2000 == 0: print('sub', i, flush=True)
    sub = pd.DataFrame(sub_rows, columns=['id', 'class', 'predicted'])
    te = pd.read_csv(f'{ROOT}/data/test.csv')
    sub = te[['id', 'class']].merge(sub, on=['id', 'class'], how='left')
    sub['predicted'] = sub['predicted'].fillna('')
    sub.to_csv(a.out_submission, index=False)
    print('WROTE', a.out_submission, 'nonempty', int((sub.predicted != '').sum()), flush=True)

if __name__ == '__main__':
    main()
