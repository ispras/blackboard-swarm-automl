import argparse, os, time, json, math, random
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data import resolve_data_config
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, 'data')

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class CassavaDS(Dataset):
    def __init__(self, df, img_dir, size, train=True, soft=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.soft = soft
        if train:
            self.tf = A.Compose([
                A.RandomResizedCrop(size, size, scale=(0.45, 1.0), ratio=(0.75, 1.333)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15, rotate_limit=25, border_mode=0, p=0.6),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.HueSaturationValue(10, 15, 10, p=0.4),
                A.CoarseDropout(max_holes=8, max_height=size//12, max_width=size//12, p=0.35),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.Resize(size, size),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = np.array(Image.open(os.path.join(self.img_dir, r.image_id)).convert('RGB'))
        x = self.tf(image=im)['image']
        if self.soft is not None:
            y = torch.tensor(self.soft[i], dtype=torch.float32)
        else:
            y = int(r.label)
        return x, y

@torch.no_grad()
def predict(model, loader, device, tta=True, nclass=5):
    model.eval()
    outs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            if tta: logits = (logits + model(torch.flip(x, dims=[3]))) / 2
        outs.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(outs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k_384')
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split', default=os.path.join(os.path.dirname(__file__),'split5_seed42.csv'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--train-all', action='store_true')
    ap.add_argument('--label-smooth', type=float, default=0.1)
    ap.add_argument('--drop-path', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--resume', default=None)
    args = ap.parse_args()
    seed_all(args.seed)
    out = args.out or os.path.join(os.path.dirname(__file__), 'runs', f"{args.model.replace('.','_')}_s{args.size}_f{args.fold}_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out,'args.json'),'w'))
    device = 'cuda'
    torch.backends.cudnn.benchmark = True

    d = pd.read_csv(args.split)
    tr = d[d.fold != args.fold].reset_index(drop=True)
    va = d[d.fold == args.fold].reset_index(drop=True)
    if args.train_all:
        tr = d.reset_index(drop=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=5, drop_path_rate=args.drop_path)
    model.to(device)
    try: model = model.to(memory_format=torch.channels_last)
    except Exception: pass

    train_ds = CassavaDS(tr, os.path.join(DATA,'train_images'), args.size, train=True)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers>0)
    if not args.train_all:
        va_ds = CassavaDS(va, os.path.join(DATA,'train_images'), args.size, train=False)
        va_loader = DataLoader(va_ds, batch_size=args.bs*2, shuffle=False, num_workers=args.workers, pin_memory=True)

    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1,len(train_loader))
    total_steps = steps_per_epoch*args.epochs
    warm_steps = steps_per_epoch*args.warmup
    def lr_lambda(step):
        if step < warm_steps: return step/max(1,warm_steps)
        p = (step-warm_steps)/max(1,total_steps-warm_steps)
        return 0.5*(1+math.cos(math.pi*p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    best = -1.0; start_ep = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler']); best = ck.get('best',-1); start_ep = ck.get('epoch',0)+1
        print('resumed from', args.resume, 'epoch', start_ep)

    logf = open(os.path.join(out,'log.txt'),'a')
    for ep in range(start_ep, args.epochs):
        model.train(); t0=time.time(); run=0.0; nb=0
        for x,y in train_loader:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); nb += 1
        msg = f"ep {ep} loss {run/max(1,nb):.4f} lr {opt.param_groups[0]['lr']:.2e} time {time.time()-t0:.0f}s"
        acc = None
        if not args.train_all:
            probs = predict(model, va_loader, device, tta=False)
            acc = (probs.argmax(1) == va.label.values).mean()
            msg += f" val_acc {acc:.5f}"
            if acc > best:
                best = acc
                torch.save({'model':model.state_dict()}, os.path.join(out,'best.pth'))
                np.save(os.path.join(out,'val_probs.npy'), probs)
                va[['image_id']].assign(pred=probs.argmax(1)).to_csv(os.path.join(out,'val_pred.csv'), index=False)
        print(msg, flush=True); logf.write(msg+'\n'); logf.flush()
        torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'sched':sched.state_dict(),
                    'scaler':scaler.state_dict(),'best':best,'epoch':ep}, os.path.join(out,'last.pth'))
    if not args.train_all:
        print('BEST', best, flush=True); logf.write(f'BEST {best}\n')
    # final: load best (or current if train_all) and predict test with tta
    if (not args.train_all) and os.path.exists(os.path.join(out,'best.pth')):
        model.load_state_dict(torch.load(os.path.join(out,'best.pth'), map_location='cpu')['model'])
    te = pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    te_ds = CassavaDS(te, os.path.join(DATA,'test_images'), args.size, train=False)
    te_loader = DataLoader(te_ds, batch_size=args.bs*2, shuffle=False, num_workers=args.workers, pin_memory=True)
    tp = predict(model, te_loader, device, tta=True)
    np.save(os.path.join(out,'test_probs.npy'), tp)
    te.assign(label=tp.argmax(1)).to_csv(os.path.join(out,'submission.csv'), index=False)
    print('DONE', out, flush=True)

if __name__ == '__main__':
    main()
