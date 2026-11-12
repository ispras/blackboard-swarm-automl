import os, sys, json, time, argparse
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
ROOT='/download_job_code/payload/sandbox'; MC=f'{ROOT}/workspaces/mimosa_sigma/cache'; CLS=['large_bowel','small_bowel','stomach']
def pp(vol,cc):
    if cc<=1 or not vol.any(): return vol
    lab,n=ndimage.label(vol,structure=np.ones((3,3,3),bool))
    if n==0: return vol
    sz=np.bincount(lab.ravel()); keep=sz>=cc; keep[0]=False
    return keep[lab]
def pts(m,cap=8000):
    a=np.argwhere(m>0).astype(np.float32)
    if len(a)>cap: a=a[np.random.RandomState(0).choice(len(a),cap,replace=False)]
    return a
def hd(p,g,tree_g,norm):
    if not p.any() and not g.any(): return 0.0
    if not p.any() or not g.any(): return 1.0
    a=pts(p); b=pts(g)
    da=cKDTree(b).query(a,k=1)[0].max()
    db=tree_g.query(a if False else None) if False else None
    # gt->pred distance
    tree_a=cKDTree(a); db=tree_a.query(b,k=1)[0].max()
    return min(1.0,float(max(da,db))/norm)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--probs',required=True); ap.add_argument('--valid',required=True); ap.add_argument('--fold',type=int,default=0); ap.add_argument('--tag',default='m'); ap.add_argument('--thr',default='0.4,0.5,0.55,0.6,0.65,0.7'); ap.add_argument('--cc',default='0,100,300')
    args=ap.parse_args(); t0=time.time()
    ids=json.load(open(f'{MC}/train_ids.json')); study=json.load(open(f'{MC}/train_study.json'))
    foldarr=np.load(f'{MC}/train_fold.npy'); mask=np.load(f'{MC}/train_mask.npy',mmap_mode='r')
    va=np.where(foldarr==args.fold)[0]
    probs=np.load(args.probs,mmap_mode='r'); valid=np.load(args.valid,allow_pickle=True); id2v={x:k for k,x in enumerate(valid)}
    groups={}
    for i in va: groups.setdefault(study[i],[]).append(i)
    order=[]; rowsof=[]
    for g in groups:
        idxs=np.array(groups[g]); rows=np.array([id2v[ids[i]] for i in idxs]); groups[g]=idxs; rowsof.append((g,rows))
    hw=(320,320); norm=np.sqrt(2)*320
    thrs=[float(x) for x in args.thr.split(',')]; ccs=[int(x) for x in args.cc.split(',')]
    # cache GT
    GT={}; GTT={}
    for g,idxs in groups.items():
        for c in range(3):
            gv=np.ascontiguousarray(mask[idxs,c]>0); GT[(g,c)]=gv; GTT[(g,c)]=cKDTree(pts(gv))
    print('gt cached',f'{time.time()-t0:.0f}s',flush=True)
    best={}
    for c in range(3):
        rows=[]
        for thr in thrs:
            for cc in ccs:
                ds=[];hs=[]
                for g,idxs in groups.items():
                    pv=np.asarray(probs[np.array([id2v[ids[i]] for i in idxs]),c])>=thr
                    if cc>1: pv=pp(pv,cc)
                    gv=GT[(g,c)]
                    inter=(pv&gv).sum(axis=(1,2)); sp=pv.sum(axis=(1,2)); sg=gv.sum(axis=(1,2))
                    den=sp+sg; d=np.where(den==0,1.0,2.0*inter/np.maximum(den,1))
                    ds.append(d.mean()); hs.append(hd(pv,gv,GTT[(g,c)],norm))
                sc=0.4*np.mean(ds)+0.6*(1-np.mean(hs))
                rows.append((float(sc),float(np.mean(ds)),float(np.mean(hs)),thr,cc))
        rows.sort(reverse=True); best[c]=rows[0]
        print(f'{CLS[c]} combo={rows[0][0]:.4f} dice={rows[0][1]:.4f} hd={rows[0][2]:.4f} thr={rows[0][3]} cc={rows[0][4]} ({time.time()-t0:.0f}s)',flush=True)
    mc=np.mean([best[c][0] for c in range(3)])
    print(f'OVERALL {args.tag} combo={mc:.4f} dice={np.mean([best[c][1] for c in range(3)]):.4f} thrs={[best[c][3] for c in range(3)]} ccs={[best[c][4] for c in range(3)]}',flush=True)
    json.dump({'tag':args.tag,'combo':float(mc),'best':{CLS[c]:{'combo':best[c][0],'dice':best[c][1],'hd':best[c][2],'thr':best[c][3],'cc':best[c][4]} for c in range(3)}},open(f'logs/evalfast_{args.tag}.json','w'),indent=2)
if __name__=='__main__': main()
