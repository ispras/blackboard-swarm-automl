import argparse, os, numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein as Lev
ROOT='/download_job_code/payload/sandbox'
FB=open(f'{ROOT}/workspaces/regulus_iota/baseline_constant.txt').read().strip()

def load_pred(path, name):
    d=pd.read_csv(path)
    col='inchi' if 'inchi' in d.columns else ('inchi_from_smiles' if 'inchi_from_smiles' in d.columns else 'InChI_pred')
    s=d[col].fillna('').astype(str).replace('nan','')
    valid=d['valid'].fillna(False).astype(bool).values if 'valid' in d.columns else (s!='').values & (s!=FB).values
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary', required=True)
    ap.add_argument('--cands', nargs='*', default=[], help='name:path')
    ap.add_argument('--q', type=float, default=0.05)
    ap.add_argument('--out', default='')
    ap.add_argument('--gt', default='')
    a=ap.parse_args()
    p=load_pred(a.primary,'primary')
    t=p
    for spec in a.cands:
        n,path=spec.split(':',1); f=load_pred(path,n); t=t.merge(f,on='image_id',how='left')
    names=['primary']+[s.split(':',1)[0] for s in a.cands]
    thr=np.quantile(t.primary_conf.values if 'primary_conf' in t.columns else t.primary.values*0+ -1, a.q) if False else None
    # derive conf from primary raw file again
    d=pd.read_csv(a.primary)
    d['image_id']=d.image_id.astype(str)
    conf_col='conf' if 'conf' in d.columns else ('confidence' if 'confidence' in d.columns else None)
    t=t.merge(d[['image_id',conf_col]].rename(columns={conf_col:'_conf'}),on='image_id',how='left')
    conf=t['_conf'].values; valid=t.primary.values!=''
    thr=np.quantile(conf,a.q)
    routed=(~valid)|(conf<=thr)
    out=[]; nr=0
    for i,r in enumerate(t.itertuples()):
        if not routed[i]:
            out.append(r.primary if r.primary else FB); continue
        nr+=1
        cands=[]
        for n in names:
            v=getattr(r,n)
            if isinstance(v,str) and v: cands.append(v)
        out.append(medoid(cands))
    res=pd.DataFrame({'image_id':t.image_id.values,'InChI':out})
    print('n',len(res),'routed',nr,'frac',round(nr/len(res),4),'conf_thr',round(float(thr),5),'fallbacks',int((res.InChI==FB).sum()))
    if a.gt:
        gt=pd.read_csv(a.gt)[['image_id','InChI']]
        mm=gt.merge(res,on='image_id',suffixes=('_gt','_pred'))
        preds=[x if isinstance(x,str) and x else FB for x in mm.InChI_pred]
        lev=np.mean([Lev.distance(x,y) for x,y in zip(preds,mm.InChI_gt)])
        # base on same rows
        basep=gt.merge(p[['image_id','primary']],on='image_id')
        bl=np.mean([Lev.distance(x if isinstance(x,str) and x else FB,y) for x,y in zip(basep.primary,basep.InChI)])
        print('VAL medoid lev',round(float(lev),4),'base',round(float(bl),4))
    if a.out: res.to_csv(a.out,index=False); print('wrote',a.out)
if __name__=='__main__': main()
