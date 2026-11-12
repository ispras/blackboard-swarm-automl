"""Fast mean Levenshtein eval for BMS. Usage:
  pyenv shell sis-black && python analysis/eval_lev.py --pred path/to/pred.csv [--truth analysis/val_holdout.csv]
pred.csv needs columns: image_id,InChI. Prints mean distance + extra stats.
"""
import argparse, pandas as pd
from rapidfuzz.distance import Levenshtein
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def dists(args):
    a,b=args
    return Levenshtein.distance(a,b)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred',required=True)
    ap.add_argument('--truth',default='analysis/val_holdout.csv')
    ap.add_argument('--show',type=int,default=5)
    a=ap.parse_args()
    truth=pd.read_csv(a.truth)
    pred=pd.read_csv(a.pred)
    m=truth.merge(pred,on='image_id',suffixes=('_t','_p'))
    miss=len(truth)-len(m)
    pairs=list(zip(m.InChI_t.tolist(), m.InChI_p.fillna('').tolist()))
    with ProcessPoolExecutor(max_workers=32) as ex:
        d=np.fromiter(ex.map(dists,pairs,chunksize=500),dtype=np.float64,count=len(pairs))
    print(f"matched={len(m)} missing={miss} mean_lev={d.mean():.4f} median={np.median(d):.2f} zero_frac={(d==0).mean():.4f} p90={np.quantile(d,.9):.1f}")
    idx=np.argsort(-d)[:a.show]
    for i in idx:
        print('--- true:',pairs[i][0][:120]); print('    pred:',pairs[i][1][:120], 'dist',d[i])
if __name__=='__main__': main()
