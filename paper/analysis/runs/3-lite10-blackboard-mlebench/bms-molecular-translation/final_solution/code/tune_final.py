"""Tune final rule on holdout. Needs paired2000.csv (1m/680k) + rigel_val20k.csv.
Prints rigel-alone on full 20k, and rule comparison + threshold sweep on the 2k paired subset."""
import argparse, pandas as pd, numpy as np, sys
sys.path.append('/usr/local/lib/python3.11/site-packages')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from rapidfuzz.distance import Levenshtein as LV
def val(s):
    try: return isinstance(s,str) and s.startswith('InChI=1S/') and s.count('/')>=2 and Chem.MolFromInchi(s) is not None
    except: return False
ap=argparse.ArgumentParser(); ap.add_argument('--paired',default='/tmp/opencode/paired2000.csv')
ap.add_argument('--rigel_val',required=True); ap.add_argument('--holdout',default='/download_job_code/payload/sandbox/analysis/holdout_20k.csv')
a=ap.parse_args()
rv=pd.read_csv(a.rigel_val); col=[c for c in ['InChI','pred','inchi'] if c in rv.columns][0]
rv=rv[['image_id',col]].rename(columns={col:'pr'})
p=pd.read_csv(a.paired); p['p1']=p['p1'].astype(str); p['p2']=p['p2'].astype(str)
m=p.merge(rv,on='image_id',how='inner'); m['pr']=m['pr'].fillna('').astype(str)
G=m['gt'].astype(str).values; p1=m['p1'].values; p2=m['p2'].values; pr=m['pr'].values
v1=m['v1'].values; v2=m['v2'].values; s1=m['s1'].values
def ev(name,pred):
    d=np.array([LV.distance(g,str(x)) for g,x in zip(G,pred)]); print(f'{name:52s} mean {d.mean():7.3f} exact {(d==0).mean():.3f}'); return d.mean()
print('=== 2k paired subset (n=%d) ==='%len(m))
gate=np.where(v1,p1,np.where(v2,p2,p1))
ev('gate (1m->680k)',gate)
ev('rigel only',pr)
ev('1m valid ? 1m : rigel',np.where(v1,p1,pr))
ev('680k valid ? 680k : rigel (ignore 1m)',np.where(v2,p2,pr))
rvv=np.array([val(x) for x in pr])
ev('rigel if valid else 1m-valid?1m:rigel',np.where(rvv,pr,np.where(v1,p1,pr)))
print('rigel valid frac',round(rvv.mean(),3),'rigel exact',round((m['pr']==m['gt']).mean(),3))
bestT=None;best=1e9
for T in [0.0,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,2.0]:
    d=ev(f'1m valid & score>={T} ? 1m : rigel',np.where(v1&(s1>=T),p1,pr))
    if d<best: best=d;bestT=T
print('BEST threshold',bestT,'mean',round(best,3))
# rigel alone on full holdout
h=pd.read_csv(a.holdout)[['image_id','InChI']].rename(columns={'InChI':'gt'})
mm=h.merge(rv,on='image_id',how='inner'); mm['pr']=mm.pr.fillna('').astype(str)
d=np.array([LV.distance(g,str(x)) for g,x in zip(mm['gt'],mm['pr'])])
print('=== rigel alone on full holdout n=%d: mean %.3f exact %.3f ==='%(len(mm),d.mean(),(d==0).mean()))
