"""Rule D final assembly.
raw-valid -> canon(raw); elif r180-valid -> canon(r180); elif flipud-valid -> canon(flipud);
elif fliplr-valid -> canon(fliplr); elif 1m-valid & score>=T -> 1m; else canon(raw).
Usage: python finalize_d.py --inchi1 preds1m_inchi.csv --rigel rigel_test_all.csv \
  --rot 'preds_rigel_rot/r180_s*.csv' --flipud 'preds_tf/t_flipud_s*.csv' --fliplr 'preds_tf/t_fliplr_s*.csv' \
  --score_thr 0.78 --out final_solution/submission.csv
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
def load_glob(pattern, name):
    fs=sorted(glob.glob(pattern))
    if not fs: return None
    d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).drop_duplicates('image_id')
    col=[c for c in ['InChI','pred','inchi'] if c in d.columns][0]
    print(name,'files',len(fs),'rows',len(d))
    return d[['image_id',col]].rename(columns={col:name})
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inchi1',default=f'{ROOT}/workspaces/altair_epsilon/preds1m_inchi.csv')
    ap.add_argument('--rigel',default=f'{ROOT}/workspaces/rigel_mu/rigel_test_all.csv')
    ap.add_argument('--rot',required=True); ap.add_argument('--flipud',required=True); ap.add_argument('--fliplr',required=True)
    ap.add_argument('--score_thr',type=float,default=0.78); ap.add_argument('--workers',type=int,default=32)
    ap.add_argument('--out',required=True); a=ap.parse_args()
    s=pd.read_csv(f'{ROOT}/data/sample_submission.csv')[['image_id']]
    d=pd.read_csv(a.inchi1)[['image_id','inchi','valid','score']].rename(columns={'inchi':'p1','valid':'v1','score':'s1'})
    r=pd.read_csv(a.rigel); rc=[c for c in ['InChI','pred','inchi'] if c in r.columns][0]
    r=r[['image_id',rc]].rename(columns={rc:'raw'}).drop_duplicates('image_id')
    parts=[s,d,r]
    for pat,name in [(a.rot,'r180'),(a.flipud,'fud'),(a.fliplr,'flr')]:
        x=load_glob(pat,name); parts.append(x)
    m=parts[0]
    for x in parts[1:]:
        if x is not None: m=m.merge(x,on='image_id',how='left')
    for c in ['p1','raw','r180','fud','flr']:
        if c not in m.columns: m[c]=''
        m[c]=m[c].fillna('').astype(str)
    m['v1']=m.v1.fillna(False).astype(bool) if 'v1' in m.columns else False
    m['s1']=m.s1.fillna(0.0).astype(float) if 's1' in m.columns else 0.0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for c,nr in [('rc','raw'),('r180c','r180'),('fudc','fud'),('flrc','flr')]:
            m[c]=list(ex.map(canon, m[nr].tolist(), chunksize=4000))
    raw_ok=np.array([valid(x) for x in m.raw.tolist()]); r_ok=np.array([valid(x) for x in m.r180.tolist()])
    fu_ok=np.array([valid(x) for x in m.fud.tolist()]); fl_ok=np.array([valid(x) for x in m.flr.tolist()])
    use1=m.v1.values&(m.s1.values>=a.score_thr)
    out=np.where(raw_ok,m.rc.values,
          np.where(r_ok,m.r180c.values,
            np.where(fu_ok,m.fudc.values,
              np.where(fl_ok,m.flrc.values,
                np.where(use1,m.p1.values,m.rc.values))))).astype(str)
    out[np.array([len(x)<6 for x in out])]='InChI=1S/'
    sub=pd.DataFrame({'image_id':m.image_id.values,'InChI':out})
    assert len(sub)==len(s) and sub.image_id.nunique()==len(s)
    os.makedirs(os.path.dirname(a.out),exist_ok=True); sub.to_csv(a.out,index=False)
    print('wrote',a.out,sub.shape,'raw_valid',int(raw_ok.sum()),'r180',int(r_ok.sum()),
          'flipud_rescue',int((fu_ok&~raw_ok&~r_ok).sum()),'fliplr_rescue',int((fl_ok&~raw_ok&~r_ok&~fu_ok).sum()),
          'use_1m',int((use1&~raw_ok&~r_ok&~fu_ok&~fl_ok).sum()),'T',a.score_thr)
if __name__=='__main__': main()
