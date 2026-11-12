import argparse, numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein as Lev
ROOT='/download_job_code/payload/sandbox'
FB=open(f'{ROOT}/workspaces/regulus_iota/baseline_constant.txt').read().strip()

def score(df, col, gt, mask=None):
    if mask is not None:
        d=df[mask]; g=[gt[i] for i in np.flatnonzero(mask)]
    else:
        d=df; g=gt
    ps=d[col].fillna(FB).tolist()
    lev=[Lev.distance(a,b) for a,b in zip(ps,g)]
    return float(np.mean(lev))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('csvs', nargs='+')
    p.add_argument('--col', default='')
    p.add_argument('--limit', type=int, default=20000)
    a=p.parse_args()
    val=pd.read_csv(f'{ROOT}/workspaces/regulus_iota/val_20k.csv')[['image_id','InChI']]
    for path in a.csvs:
        d=pd.read_csv(path)
        m=val.merge(d, on='image_id', how='inner', suffixes=('','_pred'))
        gt=m.InChI.tolist()
        col=a.col or ('inchi_from_smiles' if 'inchi_from_smiles' in d.columns else ('InChI_pred' if 'InChI_pred' in d.columns else ('inchi' if 'inchi' in d.columns else d.columns[-1])))
        if 'conf' in m.columns:
            for th in [-2.0,-1.5,-1.2,-1.0,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.25,-0.2]:
                dd=m.copy(); dd[col]=np.where(dd.conf>=th, dd[col], FB)
                ps=dd[col].fillna(FB).tolist()
                print(f'{path} th={th} n={len(m)} lev={np.mean([Lev.distance(x,y) for x,y in zip(ps,gt)]):.4f}')
        else:
            ps=m[col].fillna(FB).tolist()
            print(f'{path} col={col} n={len(m)} lev={np.mean([Lev.distance(x,y) for x,y in zip(ps,gt)]):.4f}')
