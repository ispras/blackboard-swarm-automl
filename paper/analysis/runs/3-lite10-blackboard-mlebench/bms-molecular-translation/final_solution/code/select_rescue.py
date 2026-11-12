"""After 1m shards: select rescue subset (invalid or low confidence).
Usage: python select_rescue.py --preds 'workspaces/altair_epsilon/preds/shard*.csv' --out rescue_ids.csv [--q 0.20]
Writes csv with image_id,reason; prints stats."""
import argparse, glob, sys, os
import pandas as pd, numpy as np
from concurrent.futures import ProcessPoolExecutor
ROOT='/download_job_code/payload/sandbox'
sys.path.append('/usr/local/lib/python3.11/site-packages'); sys.path.append(f'{ROOT}/workspaces/altair_epsilon')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from postproc import smi_to_inchi_fallback
def work(s):
    try: return smi_to_inchi_fallback(s)
    except Exception: return ''
def valid(s):
    if not isinstance(s,str) or not s.startswith('InChI=1S/') or s.count('/')<2: return False
    try: return Chem.MolFromInchi(s) is not None
    except Exception: return False
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preds',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--q',type=float,default=0.20); ap.add_argument('--workers',type=int,default=32)
    ap.add_argument('--invalid_only',action='store_true')
    a=ap.parse_args(); files=sorted(glob.glob(a.preds)); print('files',files)
    d=pd.concat([pd.read_csv(f) for f in files],ignore_index=True).drop_duplicates('image_id')
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        d['inchi']=list(ex.map(work,d.smiles.astype(str).tolist(),chunksize=2000))
    d['valid']=[valid(x) for x in d['inchi']]
    sc=pd.to_numeric(d.get('score',0),errors='coerce').fillna(0).values
    thr=float(np.quantile(sc,a.q)) if a.q>0 else -1
    if a.invalid_only:
        resc=d[~d.valid]
    else:
        resc=d[(~d.valid)|(sc<=thr)]
    r=pd.DataFrame({'image_id':resc.image_id,
                    'reason':np.where(~resc.valid,'invalid','lowconf')})
    r.to_csv(a.out,index=False)
    print(f'rows {len(d)} valid {d.valid.mean():.4f} invalid {int((~d.valid).sum())} score_thr(q{a.q}) {thr:.4f} rescue {len(r)} ({len(r)/max(1,len(d)):.3f})')
    print('rescue file',a.out)
if __name__=='__main__': main()
