import argparse, pandas as pd, sys
sys.path.append('/usr/local/lib/python3.11/site-packages')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
ROOT='/download_job_code/payload/sandbox'
ap=argparse.ArgumentParser(); ap.add_argument('--sub',required=True); a=ap.parse_args()
s=pd.read_csv(a.sub); ref=pd.read_csv(f'{ROOT}/data/sample_submission.csv')
print('rows',len(s),'expected',len(ref),'cols',s.columns.tolist())
assert len(s)==len(ref), 'row count mismatch'
assert s.columns.tolist()==['image_id','InChI'], s.columns.tolist()
assert s.image_id.nunique()==len(s), 'dup ids'
miss=set(ref.image_id)-set(s.image_id); extra=set(s.image_id)-set(ref.image_id)
print('missing',len(miss),'extra',len(extra))
assert not miss and not extra, (len(miss),len(extra))
inc=s.InChI.astype(str)
print('nan',s.InChI.isna().sum(),'startswith InChI=1S/',(inc.str.startswith('InChI=1S/')).mean())
print('len mean',inc.str.len().mean(),'min',inc.str.len().min(),'max',inc.str.len().max())
print('placeholder InChI=1S/',int((inc=='InChI=1S/').sum()))
def ok(x):
    try: return Chem.MolFromInchi(x) is not None
    except: return False
import numpy as np
idx=np.random.RandomState(0).choice(len(s),min(20000,len(s)),replace=False)
print('RDKit-valid frac (20k sample)',round(np.mean([ok(inc.iloc[i]) for i in idx]),4))
print('OK')
