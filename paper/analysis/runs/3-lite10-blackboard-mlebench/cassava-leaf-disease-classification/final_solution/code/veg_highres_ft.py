import argparse, os, json, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
import timm
import train as T

DATA=T.DATA

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True); ap.add_argument('--ckpt',default='best.pth')
    ap.add_argument('--size',type=int,required=True)
    ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--lr',type=float,default=4e-5)
    ap.add_argument('--wd',type=float,default=0.05); ap.add_argument('--bs',type=int,default=16)
    ap.add_argument('--workers',type=int,default=8); ap.add_argument('--out',required=True)
    ap.add_argument('--label-smooth',type=float,default=0.1)
    args=ap.parse_args()
    a=json.load(open(os.path.join(args.run,'args.json')))
    os.makedirs(args.out,exist_ok=True); json.dump(vars(args),open(os.path.join(args.out,'args.json'),'w'))
    dev='cuda'; torch.backends.cudnn.benchmark=True
    try:
        model=timm.create_model(a['model'],pretrained=False,num_classes=5,drop_path_rate=a.get('drop_path',0.0),img_size=args.size).to(dev)
    except Exception:
        model=timm.create_model(a['model'],pretrained=False,num_classes=5,drop_path_rate=a.get('drop_path',0.0)).to(dev)
    model.load_state_dict(torch.load(os.path.join(args.run,args.ckpt),map_location='cpu')['model'])
    model=model.to(memory_format=torch.channels_last)
    d=pd.read_csv(a['split']); tr=d[d.fold!=a['fold']].reset_index(drop=True); va=d[d.fold==a['fold']].reset_index(drop=True)
    tds=T.CassavaDS(tr,os.path.join(DATA,'train_images'),args.size,train=True)
    tdl=DataLoader(tds,batch_size=args.bs,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True,persistent_workers=args.workers>0)
    vds=T.CassavaDS(va,os.path.join(DATA,'train_images'),args.size,train=False)
    vdl=DataLoader(vds,batch_size=args.bs,shuffle=False,num_workers=args.workers,pin_memory=True)
    crit=nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.wd)
    total=max(1,len(tdl))*args.epochs
    def lam(s): return 0.5*(1+math.cos(math.pi*s/max(1,total)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lam); scaler=torch.cuda.amp.GradScaler()
    best=-1
    for ep in range(args.epochs):
        model.train(); t0=time.time(); run=0; nb=0
        for x,y in tdl:
            x=x.to(dev,non_blocking=True,memory_format=torch.channels_last); y=y.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): loss=crit(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); run+=loss.item(); nb+=1
        vp=T.predict(model,vdl,dev,tta=False); acc=float((vp.argmax(1)==va.label.values).mean())
        print(f'hr ep {ep} loss {run/max(1,nb):.4f} val_acc {acc:.5f} time {time.time()-t0:.0f}s',flush=True)
        if acc>best: best=acc; torch.save({'model':model.state_dict()},os.path.join(args.out,'best.pth'))
    model.load_state_dict(torch.load(os.path.join(args.out,'best.pth'),map_location='cpu')['model'])
    vp=T.predict(model,vdl,dev,tta=True); np.save(os.path.join(args.out,'val_probs_tta.npy'),vp)
    vacc=float((vp.argmax(1)==va.label.values).mean()); open(os.path.join(args.out,'val_acc.txt'),'w').write(str(vacc))
    print('hr final val_acc_tta',vacc,flush=True)
    te=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    ted=T.CassavaDS(te,os.path.join(DATA,'test_images'),args.size,train=False)
    tdl2=DataLoader(ted,batch_size=args.bs,shuffle=False,num_workers=args.workers,pin_memory=True)
    tp=T.predict(model,tdl2,dev,tta=True); np.save(os.path.join(args.out,'test_probs.npy'),tp)
    te.assign(label=tp.argmax(1)).to_csv(os.path.join(args.out,'submission.csv'),index=False)
    print('done',args.out,flush=True)

if __name__=='__main__': main()
