import pandas as pd, numpy as np
from rapidfuzz.distance import Levenshtein as Lev
ROOT='/download_job_code/payload/sandbox'
FB=open(f'{ROOT}/workspaces/regulus_iota/baseline_constant.txt').read().strip()
def load(path,name,col=None):
    d=pd.read_csv(path)
    if col is None: col='inchi' if 'inchi' in d.columns else ('inchi_from_smiles' if 'inchi_from_smiles' in d.columns else 'InChI_pred')
    s=d[col].fillna('').astype(str).replace('nan','')
    valid=d['valid'].fillna(False).astype(bool).values if 'valid' in d.columns else ((s!='').values & (s!=FB).values)
    s=np.where(valid & (s!='') & (s!=FB), s, '')
    return pd.DataFrame({'image_id':d.image_id.astype(str), name:s})
def medoid(cs):
    cs=[c for c in cs if c]
    if not cs: return FB
    if len(cs)==1: return cs[0]
    best=None; bc=None
    for c in cs:
        t=sum(Lev.distance(c,o) for o in cs)
        if best is None or t<best: best=t; bc=c
    return bc
p=load(f'{ROOT}/workspaces/regulus_iota/preds_phase3_test.csv','p')
o=load(f'{ROOT}/workspaces/altair_theta/preds_iota_test.csv','o')
r=load(f'{ROOT}/workspaces/regulus_iota/preds_phase3_rot_f8.csv','r')
om=load(f'{ROOT}/workspaces/regulus_omega/preds_omegaT_p3f8.csv','om','inchi')
m=pd.concat([load(f'{ROOT}/workspaces/altair_theta/ms_routed2.csv','m','inchi'),
             load(f'{ROOT}/workspaces/regulus_iota/ms_out/preds_ms_delta_phase3.csv','m','inchi'),
             load(f'{ROOT}/workspaces/regulus_iota/ms_out/preds_ms_f8missing.csv','m','inchi')],ignore_index=True).drop_duplicates('image_id')
f8=set(pd.read_csv(f'{ROOT}/workspaces/regulus_omega/lowconf_p3_f8.csv').image_id.astype(str))
inv=set(p[~p.p.isin([''])].image_id)  # placeholder
# add my invalid ids
dl=pd.read_csv(f'{ROOT}/workspaces/regulus_iota/preds_phase3_test.csv'); dl['valid']=dl['valid'].fillna(False).astype(bool)
invalids=set(dl.loc[~dl.valid,'image_id'].astype(str))
routed=f8|invalids
print('f8',len(f8),'invalids',len(invalids),'routed',len(routed))
t=p
for f in [o,r,om,m]: t=t.merge(f,on='image_id',how='left')
ss=pd.read_csv(f'{ROOT}/data/sample_submission.csv')
out=[]; nr=0; fb=0
for rr in t.itertuples():
    pv=rr.p if isinstance(rr.p,str) else ''
    if rr.image_id not in routed:
        out.append(pv if pv else FB); continue
    nr+=1
    cs=[pv]
    for nm in ['o','r','om','m']:
        v=getattr(rr,nm)
        if isinstance(v,str) and v: cs.append(v)
    v=medoid(cs); fb+=(v==FB); out.append(v)
res=pd.DataFrame({'image_id':t.image_id,'InChI':out})
res=ss[['image_id']].merge(res,on='image_id',how='left').fillna(FB)
res.to_csv('submission_medoid_phase3_f8.csv',index=False)
print('rows',len(res),'ids match',set(res.image_id)==set(ss.image_id),'dupes',res.image_id.duplicated().sum(),'nulls',res.InChI.isna().sum())
print('routed',nr,'fallbacks',int((res.InChI==FB).sum()),'prefix',res.InChI.str.startswith('InChI=1S/').mean())
