import argparse, json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import hd_score, points_of, CLASSES
from postproc3d import clean_volume

ROOT = "/download_job_code/payload/sandbox"
CACHE = f"{ROOT}/data/cache_train_320"

def class_score(P, gt, groups, c, thr, min_vol=0, msa=0, keep_top=0):
    pv = P[:, c] > int(thr * 255)
    g = (gt == c + 1)
    inter = np.logical_and(pv, g).reshape(len(pv), -1).sum(1)
    ssum = pv.reshape(len(pv), -1).sum(1) + g.reshape(len(g), -1).sum(1)
    d = np.where(ssum > 0, 2 * inter / np.maximum(ssum, 1), 1.0).mean()
    hds = []
    for _, ii in groups:
        p = pv[ii]
        g = (gt[ii] == c + 1)
        if min_vol or msa:
            p = clean_volume(p, min_vol=min_vol, keep_top=keep_top, min_slice_area=msa)
        hds.append(hd_score(points_of(p), points_of(g)))
    h = float(np.mean(hds))
    return float(d), h, 0.4 * float(d) + 0.6 * h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--thrs', default='0.30,0.40,0.45,0.50,0.55,0.60,0.65,0.70')
    ap.add_argument('--min_vols', default='0,50,150,400,1000')
    ap.add_argument('--min_slice_areas', default='0,30')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    index = pd.read_csv(f'{CACHE}/index.csv')
    pos = {v: i for i, v in enumerate(index['id'].values)}
    ids = pd.read_csv(a.ids)['id'].values
    rows = np.array([pos[v] for v in ids])
    vi = index.iloc[rows].reset_index(drop=True)
    gt = np.asarray(np.load(f'{CACHE}/masks_u8.npy', mmap_mode='r')[rows])
    P = np.asarray(np.load(a.probs, mmap_mode='r'))
    groups = [(cd, g.index.values) for cd, g in vi.groupby('case_day')]
    thrs = [float(x) for x in a.thrs.split(',')]
    min_vols = [int(x) for x in a.min_vols.split(',')]
    msas = [int(x) for x in a.min_slice_areas.split(',')]

    best_thr = {}
    for c in range(3):
        best = None
        for t in thrs:
            d, h, combo = class_score(P, gt, groups, c, t)
            print(f'cls={c} thr={t:.2f} dice={d:.4f} hd={h:.4f} combo={combo:.4f}', flush=True)
            if best is None or combo > best['combo']:
                best = dict(thr=t, dice=d, hd=h, combo=combo)
        best_thr[c] = best
    print('BEST_THR', json.dumps(best_thr), flush=True)

    final = {}
    for c in range(3):
        t = best_thr[c]['thr']
        best = None
        for mv in min_vols:
            for msa in msas:
                d, h, combo = class_score(P, gt, groups, c, t, mv, msa)
                if best is None or combo > best['combo']:
                    best = dict(thr=t, min_vol=mv, msa=msa, keep_top=0, dice=d, hd=h, combo=combo)
        print(f'cls={c} BEST_POST {json.dumps(best)}', flush=True)
        final[c] = best
    overall_combo = float(np.mean([v['combo'] for v in final.values()]))
    overall_dice = float(np.mean([v['dice'] for v in final.values()]))
    overall_hd = float(np.mean([v['hd'] for v in final.values()]))
    out = dict(per_class=final, overall=dict(dice=overall_dice, hd=overall_hd, combo=overall_combo))
    print('FINAL', json.dumps(out), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
