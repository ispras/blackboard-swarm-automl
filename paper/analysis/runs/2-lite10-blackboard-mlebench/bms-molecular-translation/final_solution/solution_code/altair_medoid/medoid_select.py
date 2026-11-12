import os,sys,argparse,numpy as np,pandas as pd
from rapidfuzz.distance import Levenshtein as Lev
ROOT='/download_job_code/payload/sandbox'
FB=open(f'{ROOT}/workspaces/regulus_iota/baseline_constant.txt').read().strip()

def load(p):
    df=pd.read_csv(p)
    ic=[c for c in ['inchi_from_smiles','inchi','InChI_pred','InChI'] if c in df.columns][0]
    out={}
    for i,x,c in zip(df.image_id.astype(str),df[ic].fillna(''),df['conf'] if 'conf' in df.columns else [0]*len(df)):
        xs=str(x).strip(); out[i]=('' if xs in ('','nan') or xs==FB else xs, float(c))
    return out

def medoid(c):
    if len(c)==1: return c[0]
    return min(c,key=lambda a: sum(Lev.distance(a,b) for b in c))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--iota',required=True)
    ap.add_argument('--routed',required=True)
    ap.add_argument('--cand',action='append',default=[],help='name:csv')
    ap.add_argument('--out',required=True)
    ap.add_argument('--val',default='')
    a=ap.parse_args()
    iota=load(a.iota)
    C={s.split(':',1)[0]:load(s.split(':',1)[1]) for s in a.cand}
    route=pd.read_csv(a.routed); rids=set(route.image_id.astype(str))
    ss=pd.read_csv(f'{ROOT}/data/sample_submission.csv'); ids=ss.image_id.astype(str).tolist()
    rows=[]; switched=0
    for i in ids:
        a0=iota.get(i,('',-1e9))[0]
        if i in rids:
            cands=[a0] if a0 else []
            for n,d in C.items():
                if d.get(i,('',-1e9))[0]: cands.append(d[i][0])
            p=medoid(cands) if cands else (a0 or FB)
            if p!=a0: switched+=1
            rows.append(p or FB)
        else:
            rows.append(a0 or FB)
    pd.DataFrame({'image_id':ids,'InChI':rows}).to_csv(a.out,index=False)
    print(f"wrote {a.out} rows={len(ids)} routed={len(rids)} switched={switched} fallback={sum(1 for x in rows if x==FB)}",flush=True)
    if a.val:
        v=pd.read_csv(a.val); gt=dict(zip(v.image_id.astype(str),v.InChI.astype(str)))
        # routed file may contain subset of val ids; evaluate on ids present
        evids=[i for i in ids if i in gt]
        pr=dict(zip(ids,rows))
        d=[Lev.distance(pr[i],gt[i]) for i in evids]
        print("val mean",np.mean(d),"n",len(d),flush=True)

if __name__=='__main__': main()
