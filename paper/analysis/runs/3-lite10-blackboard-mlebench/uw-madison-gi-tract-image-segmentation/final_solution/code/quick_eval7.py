import numpy as np, pandas as pd, sys
WS="/download_job_code/payload/sandbox/workspaces/capella_zeta"
gt=np.load(f"{WS}/cache/train_mask320.npy",mmap_mode='r')
idx=pd.read_csv(f"{WS}/cache/train_index.csv"); id2=dict(zip(idx.id,np.arange(len(idx))))
ids=pd.read_csv(f"{WS}/preds/mitb3_f0_val_tta4_ids.csv"); rows=np.array([id2[i] for i in ids.id])
a=np.load(f"{WS}/preds/ens6f_val_probs.npy",mmap_mode='r')  # 6-model avg
b=np.load("/download_job_code/payload/sandbox/workspaces/mirfak_sigma/oof/unetpp_b4_f0_val_probs.npy",mmap_mode='r')
N=len(ids)
def ev(mode):
    ds=np.zeros((0,3))
    acc=[]
    for st in range(0,N,512):
        en=min(st+512,N)
        if mode=='6': p=np.asarray(a[st:en],dtype=np.float32)
        else: p=(6*np.asarray(a[st:en],dtype=np.float32)+np.asarray(b[st:en],dtype=np.float32))/7
        g=np.stack([gt[rows[st:en],c] for c in range(3)],1).astype(bool)
        for t in (0.45,0.55,0.65):
            pr=p>t; row=[]
            for c in range(3):
                inter=(pr[:,c]&g[:,c]).sum((1,2)); den=pr[:,c].sum((1,2))+g[:,c].sum((1,2))
                row.append(np.where(den>0,2*inter/np.maximum(den,1),1.0).mean())
            acc.append((t,np.array(row)))
    out={}
    for t,_ in acc: out[t]=[]
    for t,row in acc: out[t].append(row)
    for t in out: out[t]=np.mean(np.array(out[t]),0)
    return out
for m in ['6','7']:
    r=ev(m); print('model',m,{t:np.round(v,4).tolist() for t,v in r.items()},'mean@.45',round(float(np.mean(r[0.45])),4),'mean@.55',round(float(np.mean(r[0.55])),4))
