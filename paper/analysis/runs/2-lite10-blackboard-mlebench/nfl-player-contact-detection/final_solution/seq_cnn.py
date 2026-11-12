import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, time, json, sys
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from postprocess_search import smooth

t0=time.time()
def log(*a): print(f'[{time.time()-t0:.1f}s]',*a,flush=True)
torch.backends.cudnn.benchmark=True
DEV='cuda'
KEYS=['game_play','nfl_player_id_1','nfl_player_id_2']
FEATS=['dist','closing_speed','rel_speed','speed1','speed2','accel1','accel2','sa1','sa2',
       'dir_diff','orient_diff','ttc','dx','dy','distance1','distance2','dist_lag1','dist_lag-1',
       'dist_rmin5','dist_rmean5','n_close1','n_close2','step',
       'x_ndist_min','x_ndist_lag1','x_ndist_lead1','x_ndist_rmin5']
HEL=['x_ndist_min','x_ndist_lag1','x_ndist_lead1','x_ndist_rmin5']
LMAX=173


def build(kind, base_path, hel_path, has_label=True):
    base_cols=[c for c in ['contact_id']+KEYS+['step','contact'] if c in pd.read_parquet(base_path, columns=None).columns] if False else None
    b=pd.read_parquet(base_path)
    cols=[c for c in FEATS if c in b.columns]
    b=b[['contact_id']+KEYS+['step']+(['contact'] if 'contact' in b.columns else [])+[c for c in cols if c!='step']].copy()
    hcols=[c for c in HEL if c in pd.read_parquet(hel_path).columns]
    h=pd.read_parquet(hel_path, columns=['contact_id']+hcols)
    b=b.merge(h,on='contact_id',how='left')
    b=b.sort_values(KEYS+['step'],kind='mergesort').reset_index(drop=True)
    gid=b.groupby(KEYS,sort=False).ngroup().values.astype(np.int64)
    n=len(b); ns=int(gid[-1])+1
    starts=np.unique(gid,return_index=True)[1]
    lengths=np.diff(np.append(starts,n))
    idxw=np.arange(n)-np.repeat(starts,lengths)
    feats=[c for c in FEATS if c in b.columns]
    C=len(feats)
    X=np.zeros((ns,LMAX,C),dtype=np.float32); M=np.zeros((ns,LMAX),dtype=np.float32)
    vals=b[feats].values.astype(np.float32)
    X[gid,idxw,:]=vals
    M[gid,idxw]=1.0
    Y=None
    if 'contact' in b.columns:
        Y=np.zeros((ns,LMAX),dtype=np.float32); Y[gid,idxw]=b.contact.values.astype(np.float32)
    meta=pd.DataFrame({'contact_id':b.contact_id.values,'game_play':b.game_play.values,
                       'step':b.step.values,'gid':gid,'idxw':idxw})
    return X,M,Y,meta,feats,b


log('loading train')
Xtr,Mtr,Ytr,meta_tr,feats,btr=build('train','train_pair.parquet','../achernar_sigma/train_helmet_features.parquet')
log('Xtr',Xtr.shape,feats)
log('loading test')
Xte,Mte,_,meta_te,_,bte=build('test','test_pair.parquet','../achernar_sigma/test_helmet_features.parquet')
log('Xte',Xte.shape)

# normalize using train valid steps
flat=Xtr[Mtr>0]
mu=np.nanmean(flat,axis=0); sd=np.nanstd(flat,axis=0); sd[sd<1e-6]=1.0
Xtr=np.nan_to_num((Xtr-mu)/sd); Xte=np.nan_to_num((Xte-mu)/sd)
Xtr[Mtr==0]=0.0; Xte[Mte==0]=0.0
log('normalized')

folds=pd.read_csv('../castor_alpha/folds.csv'); fmap=dict(zip(folds.game_play,folds.fold))
series_fold=np.array([fmap[g] for g in btr.game_play.values[np.unique(meta_tr.gid.values,return_index=True)[1]]],dtype=np.int64)
test_gpidx=np.unique(meta_te.gid.values,return_index=True)[1]
log('fold counts',np.bincount(series_fold))


class Net(nn.Module):
    def __init__(self,C,w=96):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(C,w,5,padding=2), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w,w,5,padding=4,dilation=2), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w,w,5,padding=8,dilation=4), nn.BatchNorm1d(w), nn.ReLU(),
            nn.Conv1d(w,w//2,3,padding=1), nn.BatchNorm1d(w//2), nn.ReLU(),
            nn.Conv1d(w//2,1,1))
    def forward(self,x):
        return self.net(x.transpose(1,2)).squeeze(1)


def run_fold(f):
    torch.manual_seed(100+f); np.random.seed(100+f)
    tr_idx=np.where(series_fold!=f)[0]; va_idx=np.where(series_fold==f)[0]
    Xtr_g=torch.tensor(Xtr,device=DEV); Mtr_g=torch.tensor(Mtr,device=DEV); Ytr_g=torch.tensor(Ytr,device=DEV)
    Xte_g=torch.tensor(Xte,device=DEV); Mte_g=torch.tensor(Mte,device=DEV)
    tri=torch.tensor(tr_idx,device=DEV); vai=torch.tensor(va_idx,device=DEV)
    yv=Ytr[va_idx][Mtr[va_idx]>0]; yv=yv.astype(np.int8)
    model=Net(len(feats)).to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12)
    lossf=nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(5.0,device=DEV))
    best=(-1,None,0); bad=0
    for ep in range(12):
        model.train(); perm=tri[torch.randperm(len(tri),device=DEV)]
        tot=0; cnt=0
        for i in range(0,len(perm),128):
            idx=perm[i:i+128]
            xb=Xtr_g[idx]; mb=Mtr_g[idx]; yb=Ytr_g[idx]
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                logits=model(xb)
                ls=lossf(logits,yb)
                loss=(ls*mb).sum()/mb.sum()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            tot+=loss.item(); cnt+=1
        sched.step()
        model.eval()
        with torch.no_grad():
            pv=torch.sigmoid(model(Xtr_g[vai])).cpu().numpy()
        pvv=pv[Mtr[va_idx]>0]
        auc=roc_auc_score(yv,pvv)
        # rough mcc at rate 1.15%
        thr=np.quantile(pvv,0.9885)
        pred=(pvv>=thr).astype(np.int8)
        tp=((pred==1)&(yv==1)).sum(); fp=((pred==1)&(yv==0)).sum(); fn=((pred==0)&(yv==1)).sum(); tn=((pred==0)&(yv==0)).sum()
        den=np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn)); mcc=(tp*tn-fp*fn)/den if den>0 else 0
        log(f'fold{f} ep{ep} loss{tot/cnt:.4f} auc{auc:.5f} mcc_rate{mcc:.4f}')
        if auc>best[0]:
            best=(auc,{k:v.detach().clone() for k,v in model.state_dict().items()},ep); bad=0
        else:
            bad+=1
            if bad>=3: break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        pv=torch.sigmoid(model(Xtr_g[vai])).cpu().numpy()
        pt=torch.sigmoid(model(Xte_g)).cpu().numpy()
    return va_idx,pv,pt,best[0]


oof=np.zeros((Xtr.shape[0],LMAX),dtype=np.float32)
test_pred=np.zeros((Xte.shape[0],LMAX),dtype=np.float32)
for f in range(5):
    vai,pv,pt,auc=run_fold(f)
    oof[vai]=pv; test_pred+=pt/5
    log('fold',f,'done best auc',auc)

# flatten to rows
def flatten(P,meta):
    return P[meta.gid.values, meta.idxw.values]
oof_rows=flatten(oof,meta_tr)
test_rows=flatten(test_pred,meta_te)
# collect same ordering as btr/bte (they were sorted)
out=pd.DataFrame({'contact_id':btr.contact_id.values,'game_play':btr.game_play.values,
                  'step':btr.step.values,'nfl_player_id_1':btr.nfl_player_id_1.values,
                  'nfl_player_id_2':btr.nfl_player_id_2.values,'contact':btr.contact.values,
                  'fold':btr.game_play.map(fmap).values,'prob':oof_rows})
# need nfl ids: add from btr
out.to_parquet('oof_pair_seq.parquet',index=False)
tout=pd.DataFrame({'contact_id':bte.contact_id.values,'game_play':bte.game_play.values,
                   'step':bte.step.values,'nfl_player_id_1':bte.nfl_player_id_1.values,
                   'nfl_player_id_2':bte.nfl_player_id_2.values,'prob':test_rows})
tout.to_parquet('testpred_pair_seq.parquet',index=False)

y=out.contact.values.astype(np.int8)
log('SEQ AUC',roc_auc_score(y,oof_rows))
for w in [1,3,5]:
    if w==1: p=oof_rows.astype(np.float64)
    else:
        ds,p=smooth(out,oof_rows,w); p=p.astype(np.float64)
    idx=np.argsort(-p); ys=y[idx].astype(np.float64); tp=np.cumsum(ys); fp=np.cumsum(1-ys)
    pos=ys.sum(); neg=len(ys)-pos; fn=pos-tp; tn=neg-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); mm=np.where(den>0,(tp*tn-fp*fn)/den,-1)
    k=int(np.argmax(mm))
    log('smooth',w,'mcc',mm[k],'thr',np.sort(p)[::-1][k])
