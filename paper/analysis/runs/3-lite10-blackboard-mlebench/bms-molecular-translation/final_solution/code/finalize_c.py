"""Rule C final assembly (priority: rigel-valid > rot180-valid > high-conf 1m > raw rigel).
1. raw rigel InChI RDKit-valid -> canon(raw)
2. elif rot180 InChI RDKit-valid -> canon(rot180)
3. elif 1m InChI RDKit-valid and 1m score >= T -> 1m
4. else -> canon(raw rigel) (invalid string kept)
Usage: python finalize_c.py --inchi1 preds1m_inchi.csv --rigel rigel_test_all.csv \
  --rot 'preds_rigel_rot/r180_s*.csv' --score_thr 0.78 --out final_solution/submission.csv
"""
import argparse, glob, sys, os
import pandas as pd, numpy as np
from concurrent.futures import ProcessPoolExecutor
ROOT='/download_job_code/payload/sandbox'
sys.path.append('/usr/local/lib/python3.11/site-packages')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
def canon(s):
    try:
        m=Chem.MolFromInchi(s, sanitize=False, removeHs=False)
        if m is None: return s
        r=Chem.MolToInchi(m); return r if r else s
    except Exception: return s
def valid(s):
    try: return isinstance(s,str) and s.startswith('InChI=1S/') and s.count('/')>=2 and Chem.MolFromInchi(s) is not None
    except Exception: return False
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inchi1',default=f'{ROOT}/workspaces/altair_epsilon/preds1m_inchi.csv')
    ap.add_argument('--rigel',default=f'{ROOT}/workspaces/rigel_mu/rigel_test_all.csv')
    ap.add_argument('--rot',required=True); ap.add_argument('--score_thr',type=float,default=0.78)
    ap.add_argument('--workers',type=int,default=32); ap.add_argument('--out',required=True)
    a=ap.parse_args()
    s=pd.read_csv(f'{ROOT}/data/sample_submission.csv')[['image_id']]
    d=pd.read_csv(a.inchi1)[['image_id','inchi','valid','score']].rename(columns={'inchi':'p1','valid':'v1','score':'s1'})
    r=pd.read_csv(a.rigel); rc=[c for c in ['InChI','pred','inchi'] if c in r.columns][0]
    r=r[['image_id',rc]].rename(columns={rc:'raw'}).drop_duplicates('image_id')
    rot=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.rot))],ignore_index=True).drop_duplicates('image_id')
    rc2=[c for c in ['InChI','pred','inchi'] if c in rot.columns][0]
    rot=rot[['image_id',rc2]].rename(columns={rc2:'r180'})
    m=s.merge(d,on='image_id',how='left').merge(r,on='image_id',how='left').merge(rot,on='image_id',how='left')
    for c in ['p1','raw','r180']: m[c]=m[c].fillna('').astype(str)
    m['v1']=m.v1.fillna(False).astype(bool); m['s1']=m.s1.fillna(0.0).astype(float)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        m['rc']=list(ex.map(canon, m.raw.tolist(), chunksize=4000))
        m['r180c']=list(ex.map(canon, m.r180.tolist(), chunksize=4000))
    raw_ok=np.array([valid(x) for x in m.raw.tolist()])
    r_ok=np.array([valid(x) for x in m.r180.tolist()])
    use1=(~raw_ok)&(~r_ok)&m.v1.values&(m.s1.values>=a.score_thr)
    out=np.where(raw_ok, m.rc.values,
          np.where(r_ok, m.r180c.values,
            np.where(use1, m.p1.values, m.rc.values))).astype(str)
    out[np.array([len(x)<6 for x in out])]='InChI=1S/'
    sub=pd.DataFrame({'image_id':m.image_id.values,'InChI':out})
    assert len(sub)==len(s) and sub.image_id.nunique()==len(s)
    os.makedirs(os.path.dirname(a.out),exist_ok=True); sub.to_csv(a.out,index=False)
    print('wrote',a.out,sub.shape,'raw_valid',int(raw_ok.sum()),'rot180_rescue',int(r_ok.sum()),
          'use_1m',int(use1.sum()),'raw_fallback',int((~raw_ok&~r_ok&~use1).sum()),'T',a.score_thr)
if __name__=='__main__': main()
