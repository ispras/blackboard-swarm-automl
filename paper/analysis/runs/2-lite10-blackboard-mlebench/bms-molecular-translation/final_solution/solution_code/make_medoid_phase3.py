import pandas as pd, numpy as np
from rapidfuzz.distance import Levenshtein as Lev
ROOT='/download_job_code/payload/sandbox'
FB=open(f'{ROOT}/workspaces/regulus_iota/baseline_constant.txt').read().strip()

def load(path,name,col=None):
    d=pd.read_csv(path)
    if col is None:
        col='inchi' if 'inchi' in d.columns else ('inchi_from_smiles' if 'inchi_from_smiles' in d.columns else 'InChI_pred')
    s=d[col].fillna('').astype(str).replace('nan','')
    if 'valid' in d.columns: valid=d['valid'].fillna(False).astype(bool).values
    else: valid=(s!='').values & (s!=FB).values
    s=np.where(valid & (s!='') & (s!=FB), s, '')
    return pd.DataFrame({'image_id':d.image_id.astype(str), name:s})

def medoid(cands):
    cands=[c for c in cands if c]
    if not cands: return FB
    if len(cands)==1: return cands[0]
    best=None; bc=None
    for c in cands:
        tot=sum(Lev.distance(c,o) for o in cands)
        if best is None or tot<best: best=tot; bc=c
    return bc

p=load(f'{ROOT}/workspaces/regulus_iota/preds_phase3_test.csv','p')
o=load(f'{ROOT}/workspaces/altair_theta/preds_iota_test.csv','o')
r=load(f'{ROOT}/workspaces/regulus_iota/preds_phase3_rot_routed.csv','r')
om1=load(f'{ROOT}/workspaces/regulus_omega/preds_omegaT_routed.csv','om','inchi')
om2=load(f'{ROOT}/workspaces/regulus_omega/preds_omegaT_delta_phase3.csv','om','inchi')
om=pd.concat([om1,om2],ignore_index=True).drop_duplicates('image_id')
m1=load(f'{ROOT}/workspaces/altair_theta/ms_routed2.csv','m','inchi')
m2=load(f'{ROOT}/workspaces/regulus_iota/ms_out/preds_ms_delta_phase3.csv','m','inchi')
ms=pd.concat([m1,m2],ignore_index=True).drop_duplicates('image_id')
routed=set(pd.read_csv(f'{ROOT}/workspaces/regulus_iota/lowconf_ids_phase3.csv').image_id.astype(str))
ss=pd.read_csv(f'{ROOT}/data/sample_submission.csv')
t=p
for f in [o,r,om,ms]: t=t.merge(f,on='image_id',how='left')
n_r=0; changed=0; fallbacks=0; out=[]
for r_ in t.itertuples():
    iid=r_.image_id
    pv=r_.p if isinstance(r_.p,str) else ''
    if iid not in routed:
        val=pv if pv else FB
    else:
        n_r+=1
        cands=[pv]
        for nm in ['o','r','om','m']:
            v=getattr(r_,nm)
            if isinstance(v,str) and v: cands.append(v)
        val=medoid(cands)
        fallbacks += (val==FB)
    out.append(val)
res=pd.DataFrame({'image_id':t.image_id,'InChI':out})
res=ss[['image_id']].merge(res,on='image_id',how='left').fillna(FB)
res.to_csv('submission_medoid_phase3.csv',index=False)
print('rows',len(res),'ids match',set(res.image_id)==set(ss.image_id),'dupes',res.image_id.duplicated().sum(),'nulls',res.InChI.isna().sum())
print('routed processed',n_r,'fallbacks',int(fallbacks),'prefix ok',res.InChI.str.startswith('InChI=1S/').mean())
print('const fallback total',int((res.InChI==FB).sum()))
print('saved submission_medoid_phase3.csv')
