import numpy as np, pandas as pd, os, time
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

OUT='workspaces/sirius_tau'
T=7; HH=WW=64; NF=5
FULL_CPATH=f'{OUT}/crops/crops_ground_all.npy'
TEST_CPATH=f'{OUT}/crops/crops_ground_test.npy'
MODELDIR=f'{OUT}/models_ground'

class Net(nn.Module):
    def __init__(self,inc=T*2*3):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.MaxPool2d(2))
        self.b=nn.Sequential(blk(inc,32),blk(32,64),blk(64,128),blk(128,128),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(self,x): return self.head(self.b(x)).squeeze(1)

def predict_tta(net,rows,mm,bs=1024):
    net.eval(); out=np.zeros(len(rows))
    with torch.no_grad():
        for i in range(0,len(rows),bs):
            rr=rows[i:i+bs]
            xb=np.stack([np.asarray(mm[r]).reshape(T*2*3,HH,WW).astype(np.float32)/255.0 for r in rr])
            x=torch.from_numpy(xb).cuda()
            o=(net(x)+net(x.flip(-1)))/2
            out[i:i+len(rr)]=o.cpu().numpy()
    return out

def main():
    full=pd.read_parquet(f'{OUT}/ground_all_index.parquet')
    fold=np.zeros(len(full),int)
    for k,(tri,vai) in enumerate(GroupKFold(n_splits=NF).split(full, full.contact.values, full.game_play.values)): fold[vai]=k
    MM=np.load(FULL_CPATH,mmap_mode='r'); oof=np.zeros(len(full))
    for k in range(NF):
        net=Net().cuda(); net.load_state_dict(torch.load(f'{MODELDIR}/fold{k}.pt'))
        rows=np.where(fold==k)[0]; t0=time.time()
        oof[rows]=predict_tta(net,rows,MM)
        print('fold',k,'t',round(time.time()-t0),flush=True)
    np.save(f'{OUT}/cnn_ground_oof_all_tta.npy',oof)
    y=full.contact.values
    print('TTA v2 OOF AUC',roc_auc_score(y,oof),flush=True)
    ti=pd.read_parquet(f'{OUT}/ground_test_index.parquet'); TM=np.load(TEST_CPATH,mmap_mode='r')
    rows=list(range(len(ti))); P=np.zeros(len(rows))
    for k in range(NF):
        net=Net().cuda(); net.load_state_dict(torch.load(f'{MODELDIR}/fold{k}.pt')); P+=predict_tta(net,rows,TM)
    P/=NF; np.save(f'{OUT}/cnn_ground_test_all_tta.npy',P); print('test saved',flush=True)

if __name__=='__main__': main()
