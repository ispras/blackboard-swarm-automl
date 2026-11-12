"""R2: canon(rigel), rescue rows where raw rigel is RDKit-invalid using rot180 prediction if RDKit-valid,
then override with 1m when 1m RDKit-valid and score>=T.
Usage: python finalize_r2.py --inchi1 preds1m_inchi.csv --rigel rigel_test_all.csv \
  --rot 'workspaces/altair_epsilon/preds_rigel_rot/r180_s*.csv' --score_thr 0.85 --out final_solution/submission.csv
"""
import argparse, glob, sys, os
import pandas as pd, numpy as np
from concurrent.futures import ProcessPoolExecutor
ROOT='/download_job_code/payload/sandbox'
sys.path.append('/usr/local/lib/python3.11/site-packages'); sys.path.append(f'{ROOT}/workspaces/altair_epsilon')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from postproc import smi_to_inchi_fallback
def canon(s):
    try:
        m=Chem.MolFromInchi(s, sanitize=False, removeHs=False)
        if m is None: return s
        r=Chem.MolToInchi(m); return r if r else s
    except Exception: return s
def valid(s):
    try: return isinstance(s,str) and s.startswith('InChI=1S/') and s.count('/')>=2 and Chem.MolFromInchi(s) is not None
    except Exception: return False
def smi_work(s):
    try: return smi_to_inchi_fallback(s)
    except Exception: return ''
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inchi1',default=f'{ROOT}/workspaces/altair_epsilon/preds1m_inchi.csv')
    ap.add_argument('--rigel',default=f'{ROOT}/workspaces/rigel_mu/rigel_test_all.csv')
    ap.add_argument('--rot',required=True); ap.add_argument('--score_thr',type=float,default=0.85)
    ap.add_argument('--workers',type=int,default=32); ap.add_argument('--out',required=True)
    a=ap.parse_args()
    s=pd.read_csv(f'{ROOT}/data/sample_submission.csv')[['image_id']]
    d=pd.read_csv(a.inchi1)[['image_id','inchi','valid','score']].rename(columns={'inchi':'p1','valid':'v1','score':'s1'})
    r=pd.read_csv(a.rigel); rc=[c for c in ['InChI','pred','inchi'] if c in r.columns][0]
    r=r[['image_id',rc]].rename(columns={rc:'raw'}).drop_duplicates('image_id')
    rot=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.rot))],ignore_index=True).drop_duplicates('image_id')
    rc2=[c for c in ['InChI','pred','inchi','smiles'] if c in rot.columns][0]
    rot['rinchi']=rot[rc2].astype(str)
    rot['rvalid']=[valid(x) for x in rot.rinchi]
    m=s.merge(d,on='image_id',how='left').merge(r,on='image_id',how='left').merge(
        rot[['image_id','rinchi','rvalid']],on='image_id',how='left')
    m['p1']=m.p1.fillna('').astype(str); m['raw']=m.raw.fillna('').astype(str); m['rinchi']=m.rinchi.fillna('').astype(str)
    m['v1']=m.v1.fillna(False).astype(bool); m['s1']=m.s1.fillna(0.0).astype(float)
    m['rvalid']=m.rvalid.fillna(False).astype(bool)
    # canonicalize raw rigel
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        m['rcanon']=list(ex.map(canon, m.raw.tolist(), chunksize=4000))
    raw_valid=np.array([valid(x) for x in m.raw.tolist()])
    use1=m.v1.values & (m.s1.values>=a.score_thr)
    rescue=(~use1)&(~raw_valid)&m.rvalid.values
    # rot180 canonical
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rcanon_vals=list(ex.map(canon, m.rinchi.tolist(), chunksize=4000))
    base=np.where(m.raw.str.len().values>5, m.rcanon.values, m.p1.values).astype(str)
    base[rescue]=np.array(rcanon_vals)[rescue]
    out=np.where(use1, m.p1.values, base).astype(str)
    out[np.array([len(x)<6 for x in out])]='InChI=1S/'
    sub=pd.DataFrame({'image_id':m.image_id.values,'InChI':out})
    assert len(sub)==len(s) and sub.image_id.nunique()==len(s)
    os.makedirs(os.path.dirname(a.out),exist_ok=True); sub.to_csv(a.out,index=False)
    print('wrote',a.out,sub.shape,'use_1m',int(use1.sum()),'rescue_rot180',int(rescue.sum()),
          'raw_invalid',int((~raw_valid).sum()),'rot_valid',int(m.rvalid.sum()),'thr',a.score_thr)
if __name__=='__main__': main()
