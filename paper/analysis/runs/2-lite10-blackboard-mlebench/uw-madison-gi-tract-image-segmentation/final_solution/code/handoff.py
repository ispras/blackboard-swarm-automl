import os, sys, json, argparse, numpy as np, cv2
BASE="/download_job_code/payload/sandbox"
C=f"{BASE}/workspaces/mimosa_sigma/cache"
WS=f"{BASE}/workspaces/dubhe_zeta"
def resize_stack(src, size=320, mmap=True):
    a=np.load(src, mmap_mode="r") if mmap else np.load(src)
    N,Cc,H,W=a.shape
    out=np.zeros((N,Cc,size,size),dtype=np.float16)
    for i in range(N):
        for c in range(Cc):
            out[i,c]=cv2.resize(np.asarray(a[i,c],dtype=np.float32),(size,size),interpolation=cv2.INTER_LINEAR).astype(np.float16)
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--probs",required=True); ap.add_argument("--ids",required=True)
    ap.add_argument("--split",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--size",type=int,default=320)
    args=ap.parse_args()
    ids=np.load(args.ids,allow_pickle=True).astype(str)
    if args.split=="val":
        tr_ids=json.load(open(f"{C}/train_ids.json")); fold=np.load(f"{C}/train_fold.npy")
        target=[tr_ids[i] for i in np.where(fold==0)[0]]
    else:
        target=json.load(open(f"{C}/test_ids.json"))
    pos={sid:i for i,sid in enumerate(ids)}
    missing=[t for t in target if t not in pos]
    print("n_target",len(target),"n_mine",len(ids),"missing",len(missing), missing[:3])
    arr=resize_stack(args.probs,args.size) if os.path.getsize(args.probs) else None
    out=np.zeros((len(target),3,args.size,args.size),dtype=np.float16)
    for k,t in enumerate(target):
        out[k]=arr[pos[t]]
    np.save(args.out,out); np.save(args.out.replace(".npy","_ids.npy"),np.array(target))
    print("saved",args.out,out.shape)
if __name__=="__main__": main()
