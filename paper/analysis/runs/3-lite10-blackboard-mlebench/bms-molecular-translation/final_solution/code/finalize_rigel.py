"""Final submission.
Rule (order matters):
  r = canonicalize(rigel InChI) via RDKit round-trip where MolFromInchi succeeds, else raw rigel
  out = 1m if (1m RDKit-valid and 1m score >= T) else r (if present else 1m)
Usage: python finalize_rigel.py --inchi1 preds1m_inchi.csv --rigel rigel_test_all.csv \
        --score_thr 0.85 [--no_canon] --out final_solution/submission.csv
"""
import argparse, pandas as pd, numpy as np, os, sys
from concurrent.futures import ProcessPoolExecutor
ROOT='/download_job_code/payload/sandbox'
sys.path.append('/usr/local/lib/python3.11/site-packages')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
def canon(s):
    try:
        m=Chem.MolFromInchi(s, sanitize=False, removeHs=False)
        if m is None: return s
        r=Chem.MolToInchi(m)
        return r if r else s
    except Exception:
        return s
ap=argparse.ArgumentParser()
ap.add_argument('--inchi1',default=f'{ROOT}/workspaces/altair_epsilon/preds1m_inchi.csv')
ap.add_argument('--rigel',required=True); ap.add_argument('--score_thr',type=float,default=0.85)
ap.add_argument('--no_canon',action='store_true'); ap.add_argument('--workers',type=int,default=32)
ap.add_argument('--out',required=True); a=ap.parse_args()
s=pd.read_csv(f'{ROOT}/data/sample_submission.csv')[['image_id']]
d=pd.read_csv(a.inchi1)[['image_id','inchi','valid','score']].rename(columns={'inchi':'p1','valid':'v1','score':'s1'})
r=pd.read_csv(a.rigel); col=[c for c in ['InChI','pred','inchi'] if c in r.columns][0]
r=r[['image_id',col]].rename(columns={col:'pr'}).drop_duplicates('image_id')
m=s.merge(d,on='image_id',how='left').merge(r,on='image_id',how='left')
m['p1']=m.p1.fillna('').astype(str); m['pr']=m.pr.fillna('').astype(str)
m['v1']=m.v1.fillna(False).astype(bool); m['s1']=m.s1.fillna(0.0).astype(float)
if not a.no_canon:
    vals=m['pr'].tolist()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        m['pr']=list(ex.map(canon, vals, chunksize=4000))
    changed=int((m['pr']!=pd.Series(vals)).sum()); print('canon changed',changed)
use1=m.v1.values & (m.s1.values>=a.score_thr)
have_r=(m.pr.str.len().values>5)
out=np.where(use1, m.p1.values, np.where(have_r, m.pr.values, m.p1.values)).astype(str)
out[np.array([len(x)<6 for x in out])]='InChI=1S/'
sub=pd.DataFrame({'image_id':m.image_id.values,'InChI':out})
assert len(sub)==len(s) and sub.image_id.nunique()==len(s)
os.makedirs(os.path.dirname(a.out),exist_ok=True); sub.to_csv(a.out,index=False)
print('wrote',a.out,sub.shape,'use_1m',int(use1.sum()),'use_rigel',int((~use1 & have_r).sum()),
      'rigel_missing_non1m',int((~use1 & ~have_r).sum()),'thr',a.score_thr,
      'placeholder',int((sub.InChI=='InChI=1S/').sum()))
