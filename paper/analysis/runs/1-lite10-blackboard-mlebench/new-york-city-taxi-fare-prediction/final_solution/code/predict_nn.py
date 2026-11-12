import sys, os
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT='/download_job_code/payload/sandbox'; MIM=f'{ROOT}/workspaces/mimosa_omicron'; ME=f'{ROOT}/workspaces/aldebaran_omicron'
sys.path.insert(0,MIM)
from test_features import build_test, add_counts_te
from common import load_te, FEATS_V2, TE_COLS
CONT=FEATS_V2+TE_COLS
CAT_SPECS=[("year",8),("month",13),("weekday",7),("hour",24),("pax",9),("pcell",11820),("dcell",12765)]
class Net(nn.Module):
    def __init__(self,ncont,emb_dims,hidden=(512,256,128),pdrop=0.05):
        super().__init__(); self.embs=nn.ModuleList([nn.Embedding(v,d) for (v,d) in emb_dims])
        d_in=ncont+sum(d for _,d in emb_dims); layers=[]; prev=d_in
        for h in hidden:
            layers += [nn.Linear(prev,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Dropout(pdrop)]; prev=h
        layers += [nn.Linear(prev,1)]; self.mlp=nn.Sequential(*layers)
    def forward(self,xc,cats):
        em=[e(c) for e,c in zip(self.embs,cats)]; return self.mlp(torch.cat([xc]+em,dim=1)).squeeze(1)
ck=torch.load(f'{MIM}/nn_nn_full.pt',map_location='cpu')
te=load_te(); up=te['u_pcell']; ud=te['u_dcell']
df=build_test(); df=add_counts_te(df,te)
corrupt=df['corrupt'].to_numpy().astype(bool)
X=df[CONT].to_numpy(np.float32); X=(X-ck['mu'])/ck['sd']
pc=np.clip(np.searchsorted(up,df['p_cell'].to_numpy(np.int64)),0,len(up)-1).astype(np.int32)
dc=np.clip(np.searchsorted(ud,df['d_cell'].to_numpy(np.int64)),0,len(ud)-1).astype(np.int32)
C=np.stack([df['year'].to_numpy(np.float32).astype(np.int32)-2009,
            df['month'].to_numpy(np.float32).astype(np.int32)-1,
            df['weekday'].to_numpy(np.float32).astype(np.int32),
            df['hour'].to_numpy(np.float32).astype(np.int32),
            np.clip(df['passenger_count'].to_numpy(np.float32).astype(np.int32),0,8),pc,dc],axis=1)
Xg=torch.from_numpy(X); Cg=torch.from_numpy(C.astype(np.int64))
m=Net(len(CONT),ck['emb_dims']); m.load_state_dict(ck['model']); m.eval()
with torch.no_grad():
    out=m(Xg,[Cg[:,j] for j in range(Cg.shape[1])]).numpy()
pred=(out*ck['ys']+ck['ym']).astype(np.float32)
print('nn valid mean',pred[~corrupt].mean(),'min/max',pred.min(),pred.max(),flush=True)
np.save(f'{ME}/test_pred_mimosa_nn_full.npy',pred)
