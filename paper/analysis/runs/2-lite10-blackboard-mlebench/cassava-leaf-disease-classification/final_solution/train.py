import os, sys, argparse, time, json, math
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DATA = os.path.abspath(DATA)
CLASSES = 5
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='convnext_tiny')
    p.add_argument('--img-size', type=int, default=384)
    p.add_argument('--epochs', type=int, default=18)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--folds', default='0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='runs/convnext_tiny_384')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--mixup', type=float, default=0.0)
    p.add_argument('--drop-path', type=float, default=0.1)
    p.add_argument('--resume', default='')
    p.add_argument('--init', default='')
    return p.parse_args()


class CassavaDS(Dataset):
    def __init__(self, df, img_dir, tf):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = np.array(Image.open(os.path.join(self.img_dir, r.image_id)).convert('RGB'))
        img = self.tf(image=img)['image']
        y = int(r.label) if 'label' in self.df.columns else -1
        return img, y


def build_tf(img_size, train):
    if train:
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.55, 1.0), ratio=(0.85, 1.18)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=25, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.4),
            A.CoarseDropout(max_holes=8, max_height=img_size//12, max_width=img_size//12, p=0.3),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])


def soft_ce(logits, target, smooth=0.1):
    logp = torch.log_softmax(logits, -1)
    nll = -(target * logp).sum(-1)
    return nll.mean()


def accuracy(out, y):
    return (out.argmax(1) == y).float().mean().item()


@torch.no_grad()
def predict(model, loader, tta=True):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        x = x.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(x)
            if tta:
                o = o + model(torch.flip(x, dims=[3]))
            o = o.float()
        ps.append(torch.softmax(o, 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def main():
    a = get_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))[['image_id']]
    df['fold'] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
    for f, (_, vi) in enumerate(skf.split(df, df.label)):
        df.loc[vi, 'fold'] = f

    folds = [int(x) for x in a.folds.split(',')]
    tr_tf = build_tf(a.img_size, True)
    va_tf = build_tf(a.img_size, False)
    te_ds = CassavaDS(test, os.path.join(DATA, 'test_images'), va_tf)
    te_loader = DataLoader(te_ds, batch_size=a.batch, shuffle=False,
                           num_workers=a.workers, pin_memory=True)

    for fold in folds:
        print(f'=== fold {fold} model={a.model} size={a.img_size} ===', flush=True)
        trn = df[df.fold != fold].reset_index(drop=True)
        val = df[df.fold == fold].reset_index(drop=True)
        tr_ds = CassavaDS(trn, os.path.join(DATA, 'train_images'), tr_tf)
        va_ds = CassavaDS(val, os.path.join(DATA, 'train_images'), va_tf)
        tr_loader = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, drop_last=True,
                               num_workers=a.workers, pin_memory=True, persistent_workers=True)
        va_loader = DataLoader(va_ds, batch_size=a.batch, shuffle=False,
                               num_workers=a.workers, pin_memory=True)

        model = timm.create_model(a.model, pretrained=True, num_classes=CLASSES,
                                  drop_path_rate=a.drop_path).cuda()
        init_path = a.init.replace('{fold}', str(fold)) if a.init else ''
        if init_path and os.path.exists(init_path):
            sd = torch.load(init_path, map_location='cpu')['model']
            model.load_state_dict(sd)
            print(f'init from {init_path}', flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        steps = len(tr_loader)
        total = a.epochs * steps
        warm = a.warmup * steps

        def lr_fn(s):
            if s < warm:
                return s / max(1, warm)
            p = (s - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * p))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
        start_ep = 0
        best = -1
        if a.resume and os.path.exists(a.resume):
            ck = torch.load(a.resume, map_location='cpu')
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched']); start_ep = ck['epoch'] + 1; best = ck['best']

        target = torch.zeros(a.batch, CLASSES).cuda()
        for ep in range(start_ep, a.epochs):
            model.train(); t0 = time.time(); tot_loss = 0; nb = 0
            for x, y in tr_loader:
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
                target.zero_(); target[:len(y)].scatter_(1, y.view(-1, 1), 1.0)
                if a.mixup > 0:
                    lam = np.random.beta(a.mixup, a.mixup)
                    idx = torch.randperm(x.size(0), device=x.device)
                    x = lam * x + (1 - lam) * x[idx]
                    target = lam * target + (1 - lam) * target[idx]
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x)
                    loss = soft_ce(out.float(), target[:len(y)], 0.1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step()
                tot_loss += loss.item(); nb += 1
            vp, vy = predict(model, va_loader, tta=False)
            acc = (vp.argmax(1) == vy).mean()
            print(f'ep {ep+1}/{a.epochs} loss {tot_loss/nb:.4f} val_acc {acc:.4f} '
                  f'lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.1f}s', flush=True)
            if acc > best:
                best = acc
                torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                            'sched': sched.state_dict(), 'epoch': ep, 'best': best,
                            'model_name': a.model, 'img_size': a.img_size},
                           os.path.join(a.out, f'fold{fold}.pth'))
            if ep + 1 == a.epochs:
                torch.save({'model': model.state_dict(), 'epoch': ep, 'best': best,
                            'model_name': a.model, 'img_size': a.img_size},
                           os.path.join(a.out, f'fold{fold}_last.pth'))

        ck = torch.load(os.path.join(a.out, f'fold{fold}.pth'), map_location='cpu')
        model.load_state_dict(ck['model'])
        vp, vy = predict(model, va_loader, tta=True)
        vbest = (vp.argmax(1) == vy).mean()
        np.save(os.path.join(a.out, f'val_probs_fold{fold}.npy'), vp)
        val[['image_id']].assign(p0=vp[:, 0], p1=vp[:, 1], p2=vp[:, 2], p3=vp[:, 3], p4=vp[:, 4],
                                 label=vy) .to_csv(
            os.path.join(a.out, f'val_probs_fold{fold}.csv'), index=False)
        tp, _ = predict(model, te_loader, tta=True)
        np.save(os.path.join(a.out, f'test_probs_fold{fold}.npy'), tp)
        pd.DataFrame(tp, columns=[f'p{i}' for i in range(CLASSES)]).assign(
            image_id=test.image_id.values).to_csv(
            os.path.join(a.out, f'test_probs_fold{fold}.csv'), index=False)
        with open(os.path.join(a.out, f'fold{fold}_result.json'), 'w') as f:
            json.dump({'fold': fold, 'best_tta_acc': float(vbest), 'model': a.model,
                       'img_size': a.img_size, 'epochs': a.epochs}, f)
        print(f'fold {fold} BEST_TTA_ACC {vbest:.4f}', flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
