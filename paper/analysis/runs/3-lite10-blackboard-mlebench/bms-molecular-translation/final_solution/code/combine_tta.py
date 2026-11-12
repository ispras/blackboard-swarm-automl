"""Combine rotation-TTA MolScribe predictions into a base submission.
- base: gate submission (image_id,InChI)
- tta shards: image_id,smiles,score (one rotation/ckpt per shard)
Rule: for ids present in tta, choose the RDKit-valid candidate with highest score;
      if none valid, optionally use rigel; else keep base.
Usage: python combine_tta.py --base final_solution/submission.csv --tta 'W/altair_epsilon/preds_tta/*.csv' \
   [--rigel rigel_bothinvalid.csv] --out final_solution/submission.csv
"""
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
    try: return isinstance(s,str) and s.startswith('InChI=1S/') and s.count('/')>=2 and Chem.MolFromInchi(s) is not None
    except Exception: return False
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',required=True); ap.add_argument('--tta',required=True)
    ap.add_argument('--rigel',default=''); ap.add_argument('--out',required=True); ap.add_argument('--workers',type=int,default=32)
    a=ap.parse_args()
    base=pd.read_csv(a.base)
    files=sorted(glob.glob(a.tta)); print('tta files',len(files))
    frames=[]
    for f in files:
        d=pd.read_csv(f)
        if 'smiles' not in d.columns: print('skip',f,d.columns.tolist()); continue
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            d['inchi']=list(ex.map(work,d.smiles.astype(str).tolist(),chunksize=2000))
        d['ok']=[valid(x) for x in d.inchi]
        d['score']=pd.to_numeric(d.get('score',0),errors='coerce').fillna(0.0)
        d['src']=os.path.basename(f)
        frames.append(d[['image_id','inchi','ok','score','src']])
        print(os.path.basename(f),len(d),'valid',round(d.ok.mean(),4))
    if not frames:
        print('no tta data; copying base'); base.to_csv(a.out,index=False); return
    t=pd.concat(frames,ignore_index=True)
    # choose best valid by score per image
    tv=t[t.ok].sort_values('score',ascending=False).drop_duplicates('image_id')
    best=tv.set_index('image_id')['inchi']
    m=base.set_index('image_id')
    has=best.index
    print('tta covered ids',len(has),'of base',len(m))
    cov=m.index.isin(has)
    b_inch=m.loc[cov,'InChI'].astype(str)
    # base validity
    need=[i for i in has]
    base_bad=np.array([not valid(b_inch.loc[i]) for i in need])
    rep=np.array(need)[base_bad]
    m.loc[rep,'InChI']=best.loc[rep].values
    print('replaced (base invalid & tta valid)',len(rep))
    # fill remaining invalid tta ids with rigel
    if a.rigel:
        r=pd.read_csv(a.rigel); col=[c for c in ['InChI','pred','inchi'] if c in r.columns][0]
        r=r.set_index('image_id')[col]
        have=m.index
        rem=[i for i in need if i in r.index and not valid(m.loc[i,'InChI'])]
        rv=[i for i in rem if valid(r.loc[i]) and len(str(r.loc[i]))>5]
        m.loc[rv,'InChI']=r.loc[rv].values
        print('rigel filled',len(rv))
    out=m.reset_index()[['image_id','InChI']]
    out.to_csv(a.out,index=False)
    print('wrote',a.out,out.shape,'placeholder',int((out.InChI.astype(str)=='InChI=1S/').sum()))
if __name__=='__main__': main()
