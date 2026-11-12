import argparse, os, json, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
import timm
import train as T

ROOT = T.ROOT; DATA = T.DATA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='folder with args.json + best.pth')
    ap.add_argument('--ckpt', default='best.pth')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--label-smooth', type=float, default=0.1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    a = json.load(open(os.path.join(args.run,'args.json')))
    out = args.out or os.path.join(args.run, 'full')
    os.makedirs(out, exist_ok=True)
    device='cuda'; torch.backends.cudnn.benchmark=True
    model = timm.create_model(a['model'], pretrained=False, num_classes=5, drop_path_rate=a.get('drop_path',0.0)).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run,args.ckpt), map_location='cpu')['model'])
    try: model=model.to(memory_format=torch.channels_last)
    except Exception: pass
    d = pd.read_csv(os.path.join(DATA,'train.csv'))
    ds = T.CassavaDS(d, os.path.join(DATA,'train_images'), a['size'], train=True)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers>0)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total=max(1,len(dl))*args.epochs; warm=max(1,len(dl))*max(1,args.epochs//6)
    def lam(s):
        if s<warm: return s/max(1,warm)
        return 0.5*(1+math.cos(math.pi*(s-warm)/max(1,total-warm)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lam)
    scaler=torch.cuda.amp.GradScaler()
    logf=open(os.path.join(out,'log.txt'),'w')
    for ep in range(args.epochs):
        model.train(); t0=time.time(); run=0; nb=0
        for x,y in dl:
            x=x.to(device,non_blocking=True,memory_format=torch.channels_last); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss=crit(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run+=loss.item(); nb+=1
        msg=f"full ep {ep} loss {run/max(1,nb):.4f} time {time.time()-t0:.0f}s"; print(msg,flush=True); logf.write(msg+'\n'); logf.flush()
    torch.save({'model':model.state_dict()}, os.path.join(out,'model.pth'))
    te=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    ted=T.CassavaDS(te, os.path.join(DATA,'test_images'), a['size'], train=False)
    tel=DataLoader(ted,batch_size=args.bs*2,shuffle=False,num_workers=args.workers,pin_memory=True)
    tp=T.predict(model,tel,device,tta=True)
    np.save(os.path.join(out,'test_probs.npy'),tp)
    te.assign(label=tp.argmax(1)).to_csv(os.path.join(out,'submission.csv'),index=False)
    print('done',out,flush=True)

if __name__=='__main__': main()
