import os, sys, json, time, argparse
import numpy as np, pandas as pd, cv2
from scipy import ndimage
sys.path.insert(0,'/download_job_code/payload/sandbox/workspaces/mimosa_sigma/code')
import simlib
ROOT='/download_job_code/payload/sandbox'; MC=f'{ROOT}/workspaces/mimosa_sigma/cache'; CLASSES=simlib.CLASSES
cv2.setNumThreads(4)

def postprocess(bin3d,cc_min):
    if cc_min<=1: return bin3d
    st=np.ones((3,3,3),bool); lab,n=ndimage.label(bin3d,structure=st)
    if n==0: return bin3d
    sizes=np.bincount(lab.ravel()); keep=sizes>=cc_min; keep[0]=False
    return keep[lab]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--probs',required=True); ap.add_argument('--valid',required=True); ap.add_argument('--fold',type=int,default=0); ap.add_argument('--tag',default='m')
    args=ap.parse_args(); t0=time.time()
    ids=json.load(open(f'{MC}/train_ids.json')); study=json.load(open(f'{MC}/train_study.json'))
    foldarr=np.load(f'{MC}/train_fold.npy'); mask=np.load(f'{MC}/train_mask.npy',mmap_mode='r')
    va=np.where(foldarr==args.fold)[0]
    probs=np.load(args.probs,mmap_mode='r'); valid=np.load(args.valid,allow_pickle=True)
    id2v={x:k for k,x in enumerate(valid)}
    order=[id2v[ids[i]] for i in va]
    groups={}
    for i in va: groups.setdefault(study[i],[]).append(i)
    gt={}; g2rows={g:[] for g in groups}
    for j,i in enumerate(va): g2rows[study[i]].append(order[j])
    for g,idxs in groups.items():
        idxs=np.array(idxs)
        for c in range(3): gt[(g,c)]=np.ascontiguousarray(mask[idxs,c]>0)
    pred={c:{} for c in range(3)}
    for g in groups:
        rows=np.array(g2rows[g])
        for c in range(3): pred[c][(g,c)]=np.asarray(probs[rows,c])
    import argparse as _a
    thrs=[0.45,0.5,0.55,0.6,0.65,0.7]; ccs=[0,100,300]
    best={}
    for c in range(3):
        gtc={k:v for k,v in gt.items() if k[1]==c}; rows=[]
        for thr in thrs:
            for cc in ccs:
                dices=[];hds=[]
                for key,gvol in gtc.items():
                    pvol=pred[c][key]>=thr
                    if cc>1: pvol=postprocess(pvol,cc)
                    nz=gvol.shape[0]
                    dices.append(np.mean([simlib.dice2d(pvol[z],gvol[z]) for z in range(nz)]))
                    hds.append(simlib.hd3d_norm(pvol,gvol,(320,320)))
                rows.append((float(np.mean(0.4*np.array(dices)+0.6*(1-np.array(hds)))),float(np.mean(dices)),float(np.mean(hds)),thr,cc))
        rows.sort(reverse=True); best[c]=rows[0]
        print(f'{CLASSES[c]} combo={rows[0][0]:.4f} dice={rows[0][1]:.4f} hd={rows[0][2]:.4f} thr={rows[0][3]} cc={rows[0][4]}',flush=True)
    mc=np.mean([best[c][0] for c in range(3)]); md=np.mean([best[c][1] for c in range(3)])
    print(f'OVERALL {args.tag} combo={mc:.4f} dice={md:.4f} thrs={[best[c][3] for c in range(3)]} ccs={[best[c][4] for c in range(3)]} t={time.time()-t0:.0f}s',flush=True)
    json.dump({'tag':args.tag,'combo':mc,'dice':md,'best':{CLASSES[c]:{'combo':best[c][0],'dice':best[c][1],'hd':best[c][2],'thr':best[c][3],'cc':best[c][4]} for c in range(3)}},open(f'{ROOT}/workspaces/betelgeuse_zeta/logs/eval_{args.tag}.json','w'),indent=2)
if __name__=='__main__': main()
