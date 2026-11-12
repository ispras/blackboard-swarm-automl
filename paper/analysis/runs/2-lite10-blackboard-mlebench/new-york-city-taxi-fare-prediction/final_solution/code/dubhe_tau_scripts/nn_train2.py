import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT="/download_job_code/payload/sandbox"; FEAT=f"{ROOT}/workspaces/dubhe_tau/features"; OUT=f"{ROOT}/workspaces/dubhe_tau/models"
os.makedirs(OUT,exist_ok=True)
C30=np.cos(np.pi/6); S30=np.sin(np.pi/6)
LON0,LAT0=-75.5,39.5; CELL=0.01; NX=int(round(3.0/CELL)); NY=int(round(2.5/CELL))
def augment(X):
    plon=X[:,0];plat=X[:,1];dlon=X[:,2];dlat=X[:,3];hav=X[:,12];manh=X[:,13];bearing=X[:,14]
    hour=X[:,8];weekday=X[:,10];dpc=X[:,19];ddc=X[:,20]
    pj=X[:,21];pl=X[:,23];pe=X[:,25];dj=X[:,22];dl=X[:,24];de=X[:,26]
    b=np.radians(bearing)
    extra=np.column_stack([plon*C30-plat*S30,plon*S30+plat*C30,dlon*C30-dlat*S30,dlon*S30+dlat*C30,
        np.sin(hour*np.pi/12),np.cos(hour*np.pi/12),np.sin(weekday*2*np.pi/7),np.cos(weekday*2*np.pi/7),
        (weekday>=5).astype(np.float32),np.log1p(np.maximum(hav,0)),np.log1p(np.maximum(manh,0)),
        (hav<0.1).astype(np.float32),np.sin(b),np.cos(b),
        (np.minimum(pj,dj)<2).astype(np.float32),(np.minimum(pl,dl)<2).astype(np.float32),(np.minimum(pe,de)<2).astype(np.float32),
        ddc-dpc]).astype(np.float32)
    return np.concatenate([X,extra],axis=1)
class Net(nn.Module):
    def __init__(self,nnum,hid):
        super().__init__()
        self.epu=nn.Embedding(NX*NY,16); self.edo=nn.Embedding(NX*NY,16)
        self.eyr=nn.Embedding(10,4); self.emo=nn.Embedding(12,4); self.ehr=nn.Embedding(24,8)
        self.ewd=nn.Embedding(7,4); self.epx=nn.Embedding(9,4)
        d=nnum+16*2+4+4+8+4+4
        self.net=nn.Sequential(nn.Linear(d,hid),nn.SiLU(),nn.BatchNorm1d(hid),nn.Dropout(0.05),
                               nn.Linear(hid,hid//2),nn.SiLU(),nn.BatchNorm1d(hid//2),nn.Dropout(0.05),
                               nn.Linear(hid//2,hid//4),nn.SiLU(),nn.Linear(hid//4,1))
    def forward(self,num,pu,do,yr,mo,hr,wd,px):
        e=torch.cat([num,self.epu(pu),self.edo(do),self.eyr(yr),self.emo(mo),self.ehr(hr),self.ewd(wd),self.epx(px)],1)
        return self.net(e).squeeze(1)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--nrows',type=int,default=25000000); ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--bs',type=int,default=8192); ap.add_argument('--lr',type=float,default=2e-3)
    ap.add_argument('--hid',type=int,default=1024); ap.add_argument('--tag',type=str,default='nn2')
    ap.add_argument('--seed',type=int,default=1); ap.add_argument('--smoke',type=int,default=0)
    a=ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    meta=json.load(open(f"{FEAT}/meta.json")); n=meta['n_valid']
    X=np.load(f"{FEAT}/X_full.npy",mmap_mode='r'); y=np.load(f"{FEAT}/y_full.npy",mmap_mode='r'); ho=np.load(f"{FEAT}/holdout_full.npy",mmap_mode='r')
    hold=np.where(np.asarray(ho[:n])==1)[0]; tr=np.where(np.asarray(ho[:n])==0)[0]
    rng=np.random.default_rng(a.seed)
    if a.nrows and a.nrows<len(tr): tr=rng.choice(tr,size=a.nrows,replace=False)
    tr.sort()
    TE=[np.load(f"{FEAT}/{f}",mmap_mode='r') for f in ['TE_dec2.npy','TE_extra.npy','TE_res.npy']]
    def build(idx):
        A=augment(np.asarray(X[idx],dtype=np.float32))
        T=np.concatenate([np.asarray(t[idx],dtype=np.float32) for t in TE],axis=1)
        return np.concatenate([A,T],axis=1)
    Xtr=build(tr); Xva=build(hold); ytr=np.asarray(y[tr],dtype=np.float32); yva=np.asarray(y[hold],dtype=np.float32)
    print("loaded",Xtr.shape,Xva.shape,flush=True)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-6]=1
    Xtr-=mu; Xtr/=sd; Xva-=mu; Xva/=sd
    # raw coords for cells: recover from standardized
    def cells_raw(A):
        lon=A[:,0]*sd[0]+mu[0]; lat=A[:,1]*sd[1]+mu[1]; dlon=A[:,2]*sd[2]+mu[2]; dlat=A[:,3]*sd[3]+mu[3]
        ix=np.clip(((lon-LON0)/CELL).astype(np.int64),0,NX-1); iy=np.clip(((lat-LAT0)/CELL).astype(np.int64),0,NY-1)
        jx=np.clip(((dlon-LON0)/CELL).astype(np.int64),0,NX-1); jy=np.clip(((dlat-LAT0)/CELL).astype(np.int64),0,NY-1)
        return iy*NX+ix, jy*NX+jx
    def small_cats(A):
        raw=lambda j: A[:,j]*sd[j]+mu[j]
        yr=np.clip(raw(5).astype(np.int64)-2009,0,9); mo=np.clip(raw(6).astype(np.int64)-1,0,11)
        hr=np.clip(raw(8).astype(np.int64),0,23); wd=np.clip(raw(10).astype(np.int64),0,6)
        px=np.clip(np.nan_to_num(raw(4),nan=1).astype(np.int64),0,8)
        return yr,mo,hr,wd,px
    def pack(A):
        pu,do=cells_raw(A); yr,mo,hr,wd,px=small_cats(A)
        return [torch.tensor(A,dtype=torch.float32),torch.tensor(pu),torch.tensor(do),torch.tensor(yr),
                torch.tensor(mo),torch.tensor(hr),torch.tensor(wd),torch.tensor(px)]
    dev='cuda'; Ttr=[t.to(dev) for t in pack(Xtr)]; Tva=[t.to(dev) for t in pack(Xva)]
    ytr_t=torch.tensor(ytr/10.0,dtype=torch.float32,device=dev)
    model=Net(Xtr.shape[1],a.hid).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    nb=len(ytr)//a.bs
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=max(a.epochs*nb,1),pct_start=0.15)
    lossf=nn.MSELoss()
    def evaluate():
        model.eval(); ps=[]
        with torch.no_grad():
            for s in range(0,len(yva),32768):
                ps.append(model(*[t[s:s+32768] for t in Tva]).cpu().numpy())
        p=np.concatenate(ps)*10.0
        return float(np.sqrt(np.mean((p-yva)**2))),p
    best=1e9; bestp=None
    for ep in range(a.epochs):
        model.train(); perm=torch.randperm(len(ytr_t),device=dev); t0=time.time(); tot=0
        nsteps=nb if not a.smoke else min(nb,50)
        for it in range(nsteps):
            ii=perm[it*a.bs:(it+1)*a.bs]; opt.zero_grad()
            loss=lossf(model(*[t[ii] for t in Ttr]),ytr_t[ii]); loss.backward(); opt.step(); sched.step(); tot+=float(loss)
        r,p=evaluate(); print(f"epoch {ep} loss {tot/max(nsteps,1):.4f} val {r:.5f} {time.time()-t0:.0f}s",flush=True)
        if r<best: best=r; bestp=p.copy(); torch.save(model.state_dict(),f"{OUT}/{a.tag}.pt")
    print("BEST",best,flush=True)
    Xte=augment(np.load(f"{FEAT}/test_X.npy")); Tte=np.concatenate([np.load(f"{FEAT}/{f}") for f in ['test_TE_dec2.npy','test_TE_extra.npy','test_TE_res.npy']],axis=1)
    Xte=np.concatenate([Xte,Tte],axis=1); Xte-=mu; Xte/=sd
    model.load_state_dict(torch.load(f"{OUT}/{a.tag}.pt")); model.eval(); ps=[]
    with torch.no_grad():
        T=[t.to(dev) for t in pack(Xte)]
        for s in range(0,len(Xte),32768): ps.append(model(*[t[s:s+32768] for t in T]).cpu().numpy())
    pte=np.concatenate(ps)*10.0
    pd.DataFrame({'key':pd.read_csv(f"{FEAT}/test_keys.csv")['key'],'fare_amount':pte}).to_csv(f"{OUT}/{a.tag}_test.csv",index=False)
    np.save(f"{OUT}/{a.tag}_valpred.npy",bestp.astype(np.float32)); np.save(f"{OUT}/{a.tag}_validx.npy",hold.astype(np.int64))
    json.dump({'tag':a.tag,'val_rmse':best,'nrows':int(len(ytr)),'args':vars(a)},open(f"{OUT}/{a.tag}_result.json","w"),indent=2)
    print("SAVED",flush=True)
if __name__=='__main__': main()
