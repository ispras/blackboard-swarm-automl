import os, sys, time, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')
FEATS=['dist','closing','rel_speed','speed1','speed2','acceleration1','sa1','p1_toward','p2_toward',
       'hdist_e','hiou_e','hdist_s','hiou_s','dist_min_w5','same_team',
       'cy_e1','top_e1','height_e1','cx_e1','cy_s1','top_s1','height_s1','cx_s1',
       'nn_dist1','cnt21','cnt51','dmin_w10','cnn']
L=21; HALF=L//2

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def smooth(df,w):
    df=df.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    r=df.groupby(['game_play','p1','p2'],sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    df['sm']=r.reset_index(level=[0,1,2],drop=True).reindex(df.index).values
    return df

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
    return idx,X

class Net(nn.Module):
    def __init__(self,F,hidden=64,ns=1):
        super().__init__()
        self.gru=nn.GRU(F,hidden,batch_first=True,bidirectional=True,num_layers=1)
        self.head=nn.Sequential(nn.Linear(2*hidden*2+ns,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,1))
    def forward(self,x,s):
        h,_=self.gru(x)
        a=torch.cat([h.mean(dim=1),h.max(dim=1).values,s],dim=1)
        return self.head(a).squeeze(1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--folds',type=int,default=5); ap.add_argument('--epochs',type=int,default=16)
    ap.add_argument('--neg_frac',type=float,default=0.25); ap.add_argument('--bs',type=int,default=2048)
    ap.add_argument('--tag',default='gru'); ap.add_argument('--only_fold',type=int,default=-1)
    a=ap.parse_args(); t0=time.time()
    d=pd.read_parquet(os.path.join(CACHE,'featg_all_train.parquet'),
                      columns=['contact_id','game_play','p1','p2','step','contact','is_ground']+[c for c in FEATS if c!='cnn'])
    cn=pd.read_parquet(os.path.join(HERE,'..','sirius_tau','oof_pred.parquet')).rename(columns={'prob':'cnn'})
    d=d.merge(cn,on='contact_id',how='left')
    d=d.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    y=d.contact.values.astype(np.float32); groups=d.game_play.values
    idx,X=build_windows(d); S=np.stack([d.is_ground.values.astype(np.float32)],axis=1)
    print('windows',X.shape,'t',round(time.time()-t0),flush=True)
    dev='cuda'; oof=np.zeros(len(d)); rng=np.random.RandomState(1)
    gkf=GroupKFold(n_splits=a.folds)
    folds=range(a.folds) if a.only_fold<0 else [a.only_fold]
    for k,(tri,vai) in enumerate(gkf.split(d,y,groups)):
        if k not in folds: continue
        trp=tri[y[tri]==1]; trn=tri[y[tri]==0]; trn=trn[rng.rand(len(trn))<a.neg_frac]
        tri2=np.concatenate([trp,trn]); rng.shuffle(tri2)
        net=Net(len(FEATS)).to(dev)
        opt=torch.optim.AdamW(net.parameters(),lr=1.2e-3,weight_decay=1e-5)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
        lossf=nn.BCEWithLogitsLoss()
        vidx=vai[rng.rand(len(vai))<0.25]
        Xv=torch.tensor(X[idx[vidx]],device=dev); Sv=torch.tensor(S[vidx],device=dev); yv=y[vidx]
        best_auc=0; best_state=None; bad=0
        for ep in range(a.epochs):
            net.train(); rng.shuffle(tri2)
            for i in range(0,len(tri2),a.bs):
                b=tri2[i:i+a.bs]
                xb=torch.tensor(X[idx[b]],device=dev); sb=torch.tensor(S[b],device=dev); yb=torch.tensor(y[b],device=dev)
                opt.zero_grad(); out=net(xb,sb); l=lossf(out,yb); l.backward(); opt.step()
            sched.step(); net.eval()
            with torch.no_grad():
                pv=[]
                for i in range(0,len(vidx),16384): pv.append(net(Xv[i:i+16384],Sv[i:i+16384]).sigmoid().cpu().numpy())
                pv=np.concatenate(pv)
            auc=roc_auc_score(yv,pv); print('fold %d ep %d auc %.5f t=%d'%(k,ep,auc,time.time()-t0),flush=True)
            if auc>best_auc+1e-5: best_auc=auc; best_state={kk:v.clone() for kk,v in net.state_dict().items()}; bad=0
            else:
                bad+=1
                if bad>=4: break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            po=[]
            for i in range(0,len(vai),16384):
                b=vai[i:i+16384]; po.append(net(torch.tensor(X[idx[b]],device=dev),torch.tensor(S[b],device=dev)).sigmoid().cpu().numpy())
            oof[vai]=np.concatenate(po)
        torch.save(best_state,os.path.join(CACHE,f'{a.tag}_{k}.pt'))
        print('fold %d done best_auc %.5f t=%d'%(k,best_auc,time.time()-t0),flush=True); del net; torch.cuda.empty_cache()
    np.save(os.path.join(CACHE,f'oof_{a.tag}.npy'),oof.astype(np.float32))
    np.save(os.path.join(CACHE,f'ids_{a.tag}.npy'),d.contact_id.values)
    if a.only_fold<0:
        dt=pd.read_parquet(os.path.join(CACHE,'featg_all_test.parquet'),
                           columns=['contact_id','game_play','p1','p2','step','is_ground']+[c for c in FEATS if c!='cnn'])
        cnt=pd.read_parquet(os.path.join(HERE,'..','sirius_tau','test_pred.parquet')).rename(columns={'prob':'cnn'})
        dt=dt.merge(cnt,on='contact_id',how='left')
        dt=dt.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
        idx_t,Xt=build_windows(dt); St=np.stack([dt.is_ground.values.astype(np.float32)],axis=1)
        pt=np.zeros(len(dt))
        for k in range(a.folds):
            net=Net(len(FEATS)).to(dev); net.load_state_dict(torch.load(os.path.join(CACHE,f'{a.tag}_{k}.pt'))); net.eval()
            with torch.no_grad():
                po=[]
                for i in range(0,len(dt),16384):
                    b=slice(i,i+16384); po.append(net(torch.tensor(Xt[idx_t[b]],device=dev),torch.tensor(St[b],device=dev)).sigmoid().cpu().numpy())
                pt+=np.concatenate(po)/a.folds
        np.save(os.path.join(CACHE,f'ptest_{a.tag}.npy'),pt.astype(np.float32))
        np.save(os.path.join(CACHE,f'pids_{a.tag}.npy'),dt.contact_id.values)
        pd.DataFrame({'contact_id':d.contact_id.values,'prob':oof}).to_parquet(os.path.join(HERE,f'oof_pred_{a.tag}.parquet'))
        pd.DataFrame({'contact_id':dt.contact_id.values,'prob':pt}).to_parquet(os.path.join(HERE,f'test_pred_{a.tag}.parquet'))
        print('saved test',len(dt),flush=True)
    raw,thr=best_mcc(y,oof); tmp=d[['game_play','p1','p2','step']].copy(); tmp['prob']=oof; tmp['contact']=y
    for w in (3,5,7):
        s=smooth(tmp.copy(),w); m,t=best_mcc(s.contact.values,s.sm.values); print('smooth w%d %.5f'%(w,m),flush=True)
    print('GRU raw %.5f thr %.4f'%(raw,thr),flush=True)

if __name__=='__main__':
    main()
