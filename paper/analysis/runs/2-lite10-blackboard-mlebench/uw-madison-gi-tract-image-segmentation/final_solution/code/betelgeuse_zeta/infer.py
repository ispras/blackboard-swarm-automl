import os, sys, argparse, time, json
import numpy as np, pandas as pd, cv2, torch
import segmentation_models_pytorch as smp
cv2.setNumThreads(1)
ROOT=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(ROOT,'cache384')
MIM=320
MIM_IDS='/download_job_code/payload/sandbox/workspaces/mimosa_sigma/cache/test_ids.json'

def build(a):
    return getattr(smp,a['arch'])(encoder_name=a['encoder'],encoder_weights=None,in_channels=3,classes=3)

def load_model(ckpt,dev):
    st=torch.load(ckpt,map_location='cpu'); m=build(st['args']).to(dev); m.load_state_dict(st['model']); m.eval(); return m,st['args']

@torch.no_grad()
def predict(model, index, imgs_path, dev, bs=48, tta=True, out_size=MIM):
    imgs=np.load(imgs_path,mmap_mode='r'); idx=index.reset_index(drop=True); N=len(idx)
    out=np.zeros((N,3,out_size,out_size),np.float16); t0=time.time()
    for s in range(0,N,bs):
        rows=idx.iloc[s:s+bs]
        rr=np.stack([rows['prev'].values,rows['row'].values,rows['next'].values],axis=1)
        a=np.stack([imgs[rr[:,k]] for k in range(3)],axis=1).astype(np.float32)/255.0
        a=(a-0.5)/0.5; x=torch.from_numpy(a).to(dev)
        with torch.autocast('cuda',dtype=torch.float16):
            p=torch.sigmoid(model(x).float())
            if tta: p=(p+torch.sigmoid(model(torch.flip(x,dims=[3])).float()).flip(3))/2
        p=p.cpu().numpy().astype(np.float32)
        if p.shape[-1]!=out_size:
            p=np.stack([[cv2.resize(p[j,c],(out_size,out_size),interpolation=cv2.INTER_LINEAR) for c in range(3)] for j in range(len(rows))])
        out[s:s+len(rows)]=p.astype(np.float16)
        if s%(bs*20)==0: print(f'{s}/{N} {time.time()-t0:.0f}s',flush=True)
    return out,idx

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=True); ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--tag',required=True); ap.add_argument('--val',action='store_true'); ap.add_argument('--test',action='store_true')
    ap.add_argument('--no_tta',action='store_true'); args=ap.parse_args(); dev='cuda'
    model,ma=load_model(args.ckpt,dev)
    if args.val:
        idx=pd.read_csv(os.path.join(CACHE,'train_index.csv')); idx=idx[idx.fold==args.fold].reset_index(drop=True)
        out,idx=predict(model,idx,os.path.join(CACHE,'train_imgs.npy'),dev,tta=not args.no_tta)
        np.save(os.path.join(ROOT,args.tag+'_valprobs.npy'),out); np.save(os.path.join(ROOT,args.tag+'_valid.npy'),idx['id'].values)
        print('saved val',out.shape,flush=True)
    if args.test:
        idx=pd.read_csv(os.path.join(CACHE,'test_index.csv')).reset_index(drop=True)
        out,idx=predict(model,idx,os.path.join(CACHE,'test_imgs.npy'),dev,tta=not args.no_tta)
        order=[int(i) for i in idx['row'].values]
        # reorder to mimosa canonical test order if available
        if os.path.exists(MIM_IDS):
            ids=json.load(open(MIM_IDS)); pos={x:k for k,x in enumerate(idx['id'].values)}
            perm=np.array([pos[x] for x in ids]); out=out[perm]; idx=idx.iloc[perm].reset_index(drop=True); print('reordered to mimosa ids',flush=True)
        np.save(os.path.join(ROOT,args.tag+'_testprobs.npy'),out); np.save(os.path.join(ROOT,args.tag+'_test_ids.npy'),idx['id'].values)
        idx.to_csv(os.path.join(ROOT,args.tag+'_testidx.csv'),index=False)
        print('saved test',out.shape,flush=True)
