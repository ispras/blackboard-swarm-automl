import os, sys, time, json, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')

FEATS=['dist','closing','rel_speed','speed1','speed2','acceleration1','sa1','p1_toward','p2_toward',
       'hdist_e','hiou_e','hdist_s','hiou_s',
       'cy_e1','top_e1','height_e1','cy_s1','top_s1','height_s1',
       'nn_dist1','cnt21','cnt51','dmin_w5','cnn']
L=13; HALF=L//2

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def smooth(tmp,w):
    r=tmp.groupby(['game_play','p1','p2'],sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    return r.reset_index(level=[0,1,2],drop=True).reindex(tmp.index).values

def build_windows(d):
    keys=d[['game_play','p1','p2']]
    ch=(keys!=keys.shift()).any(axis=1).values; ch[0]=True
    starts=np.where(ch)[0]; gid=np.cumsum(ch)-1
    ends=np.empty_like(starts); ends[:-1]=starts[1:]-1; ends[-1]=len(d)-1
    N=len(d); off=np.arange(-HALF,HALF+1)
    base=(np.arange(N)[:,None]+off[None,:])
    lo=starts[gid][:,None]; hi=ends[gid][:,None]
    idx=np.clip(base,lo,hi).astype(np.int64)
    X=d[FEATS].values.astype(np.float32)
    mu=np.nanmean(X,axis=0); sd=np.nanstd(X,axis=0)+1e-6
    X=(X-mu)/sd; X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
    return idx, X

class Net(nn.Module):
    def __init__(self,F,hidden=96,ns=1):
        super().__init__()
        self.body=nn.Sequential(
            nn.Conv1d(F,hidden,3,padding=1),nn.BatchNorm1d(hidden),nn.ReLU(),
            nn.Conv1d(hidden,hidden,3,padding=1),nn.BatchNorm1d(hidden),nn.ReLU(),
            nn.Conv1d(hidden,hidden,3,padding=1),nn.BatchNorm1d(hidden),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(2*hidden+ns,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,1))
    def forward(self,x,s):
        h=self.body(x.transpose(1,2))
        a=torch.cat([h.mean(dim=2),h.max(dim=2).values,s],dim=1)
        return self.head(a).squeeze(1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--folds',type=int,default=5); ap.add_argument('--epochs',type=int,default=18)
    ap.add_argument('--neg_frac',type=float,default=0.28); ap.add_argument('--bs',type=int,default=4096)
    ap.add_argument('--tag',default='seq'); ap.add_argument('--only_fold',type=int,default=-1)
    a=ap.parse_args()
    t0=time.time()
    d=pd.read_parquet(os.path.join(CACHE,'featg_all_train.parquet'),
                      columns=['contact_id','game_play','p1','p2','step','contact','is_ground']+[c for c in FEATS if c!='cnn'])
    cn=pd.read_parquet(os.path.join(HERE,'..','sirius_tau','oof_pred.parquet')).rename(columns={'prob':'cnn'})
    d=d.merge(cn,on='contact_id',how='left')
    d=d.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    y=d.contact.values.astype(np.float32); groups=d.game_play.values
    idx,X=build_windows(d)
    S=np.stack([d.is_ground.values.astype(np.float32)],axis=1)
    print('windows built',X.shape,idx.shape,'t',round(time.time()-t0),flush=True)
    dev='cuda'
    gkf=GroupKFold(n_splits=a.folds)
    oof=np.zeros(len(d))
    folds=range(a.folds) if a.only_fold<0 else [a.only_fold]
    rng=np.random.RandomState(0)
    for k,(tri,vai) in enumerate(gkf.split(d,y,groups)):
        if k not in folds: continue
        trp=tri[y[tri]==1]; trn=tri[y[tri]==0]
        trn=trn[rng.rand(len(trn))<a.neg_frac]
        tri2=np.concatenate([trp,trn]); rng.shuffle(tri2)
        net=Net(len(FEATS),96,1).to(dev)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
        lossf=nn.BCEWithLogitsLoss()
        # val subset for early stop
        vidx=vai[rng.rand(len(vai))<0.25]
        Xv=torch.tensor(X[idx[vidx]],device=dev); Sv=torch.tensor(S[vidx],device=dev); yv=y[vidx]
        best_auc=0; best_state=None; bad=0
        for ep in range(a.epochs):
            net.train(); rng.shuffle(tri2)
            for i in range(0,len(tri2),a.bs):
                b=tri2[i:i+a.bs]
                xb=torch.tensor(X[idx[b]],device=dev); sb=torch.tensor(S[b],device=dev)
                yb=torch.tensor(y[b],device=dev)
                opt.zero_grad(); out=net(xb,sb); l=lossf(out,yb); l.backward(); opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                pv=[]
                for i in range(0,len(vidx),16384):
                    pv.append(net(Xv[i:i+16384],Sv[i:i+16384]).sigmoid().cpu().numpy())
                pv=np.concatenate(pv)
            auc=roc_auc_score(yv,pv)
            print('fold %d ep %d auc %.5f t=%d'%(k,ep,auc,time.time()-t0),flush=True)
            if auc>best_auc+1e-5:
                best_auc=auc; best_state={kk:v.clone() for kk,v in net.state_dict().items()}; bad=0
            else:
                bad+=1
                if bad>=4: break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            po=[]
            for i in range(0,len(vai),16384):
                b=vai[i:i+16384]
                po.append(net(torch.tensor(X[idx[b]],device=dev),torch.tensor(S[b],device=dev)).sigmoid().cpu().numpy())
            oof[vai]=np.concatenate(po)
        print('fold %d done best_auc %.5f t=%d'%(k,best_auc,time.time()-t0),flush=True)
        torch.save(best_state,os.path.join(CACHE,f'{a.tag}_{k}.pt'))
        del net; torch.cuda.empty_cache()
    np.save(os.path.join(CACHE,f'oof_{a.tag}.npy'),oof.astype(np.float32))
    np.save(os.path.join(CACHE,f'ids_{a.tag}.npy'),d.contact_id.values)
    # test inference
    if a.only_fold<0:
        dt=pd.read_parquet(os.path.join(CACHE,'featg_all_test.parquet'),
                           columns=['contact_id','game_play','p1','p2','step','is_ground']+[c for c in FEATS if c!='cnn'])
        cnt=pd.read_parquet(os.path.join(HERE,'..','sirius_tau','test_pred.parquet')).rename(columns={'prob':'cnn'})
        dt=dt.merge(cnt,on='contact_id',how='left')
        dt=dt.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
        idx_t,Xt=build_windows(dt)
        St=np.stack([dt.is_ground.values.astype(np.float32)],axis=1)
        pt=np.zeros(len(dt))
        for k in range(a.folds):
            net=Net(len(FEATS),96,1).to(dev); net.load_state_dict(torch.load(os.path.join(CACHE,f'{a.tag}_{k}.pt')))
            net.eval()
            with torch.no_grad():
                po=[]
                for i in range(0,len(dt),16384):
                    b=slice(i,i+16384)
                    po.append(net(torch.tensor(Xt[idx_t[b]],device=dev),torch.tensor(St[b],device=dev)).sigmoid().cpu().numpy())
                pt+=np.concatenate(po)/a.folds
        np.save(os.path.join(CACHE,f'ptest_{a.tag}.npy'),pt.astype(np.float32))
        np.save(os.path.join(CACHE,f'pids_{a.tag}.npy'),dt.contact_id.values)
        print('saved test',len(dt),flush=True)
    mask=oof>0
    if mask.sum()>0:
        raw,thr=best_mcc(y[mask],oof[mask])
        tmp=d.loc[mask,['game_play','p1','p2','step']].copy(); tmp['prob']=oof[mask]; tmp['contact']=y[mask]
        tmp=tmp.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
        best=(-1,0,0)
        for w in (1,3,5,7):
            pp=smooth(tmp,w); mm,tt=best_mcc(tmp.contact.values,pp)
            if mm>best[0]: best=(mm,w,tt)
            print('partial smooth w%d %.5f'%(w,mm),flush=True)
        print('SEQ partial raw %.5f thr %.4f BEST smooth %.5f w%d'%(raw,thr,best[0],best[1]),flush=True)

if __name__=='__main__':
    main()
