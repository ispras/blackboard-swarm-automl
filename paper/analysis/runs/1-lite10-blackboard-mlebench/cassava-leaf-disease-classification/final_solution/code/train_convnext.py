import os, sys, time, math, argparse, json, random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score

cv2.setNumThreads(0)
ROOT = "/download_job_code/payload/sandbox"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', default='convnext_small.fb_in22k_ft_in1k')
    p.add_argument('--size', type=int, default=384)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wd', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--split', default=f'{ROOT}/workspaces/shared/val_split_seed42.csv')
    p.add_argument('--images', default=f'{ROOT}/data/train_images')
    p.add_argument('--test_images', default=f'{ROOT}/data/test_images')
    p.add_argument('--sample_sub', default=f'{ROOT}/data/sample_submission.csv')
    p.add_argument('--out', default=f'{ROOT}/workspaces/achernar_delta/run')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--folds', type=str, default='0')  # e.g. "0,1,2,3,4"
    p.add_argument('--label_smooth', type=float, default=0.1)
    p.add_argument('--mixup', type=float, default=0.0)
    p.add_argument('--warmup', type=float, default=0.1)
    p.add_argument('--drop_path', type=float, default=0.1)
    p.add_argument('--no_tta', action='store_true')
    p.add_argument('--full', action='store_true', help='train on 100% of train, no val')
    return p.parse_args()


class CassavaDS(Dataset):
    def __init__(self, df, img_dir, tfm, train=True):
        self.df = df.reset_index(drop=True)
        self.dir = img_dir
        self.tfm = tfm
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(os.path.join(self.dir, r.image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(image=img)['image']
        if self.train:
            return img, int(r.label)
        return img, r.image_id


def build_tfms(size, train):
    if train:
        return A.Compose([
            A.RandomResizedCrop(size, size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=size // 20, max_width=size // 20, p=0.3),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(int(size * 1.14), int(size * 1.14)),
        A.CenterCrop(size, size),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])


def mixup_cutmix(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    if np.random.rand() < 0.5:
        x = lam * x + (1 - lam) * x2
    else:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, y[idx], lam


def rand_bbox(size, lam):
    W = size[2]; H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W); bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W); bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


@torch.no_grad()
def predict_probs(model, loader, device, tta=True):
    model.eval(); out = []; ids = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            logits = model(x)
            if tta:
                logits = logits + model(torch.flip(x, dims=[3]))
                logits = logits + model(torch.flip(x, dims=[2]))
        out.append(F.softmax(logits.float(), 1).cpu().numpy())
        ids.extend(list(y))
    return np.concatenate(out), ids


def run_fold(a, fold, df, device):
    set_seed(a.seed + fold)
    tr = df[df.fold != fold].reset_index(drop=True)
    va = df[df.fold == fold].reset_index(drop=True)
    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    tr_dl = DataLoader(CassavaDS(tr, a.images, build_tfms(a.size, True), True),
                       batch_size=a.bs, shuffle=True, drop_last=True, **dl_kw)
    va_dl = DataLoader(CassavaDS(va, a.images, build_tfms(a.size, False), True),
                       batch_size=a.bs, shuffle=False, **dl_kw)

    kwargs = {}
    try:
        model = timm.create_model(a.backbone, pretrained=True, num_classes=5, drop_path_rate=a.drop_path)
    except TypeError:
        model = timm.create_model(a.backbone, pretrained=True, num_classes=5)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(tr_dl) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=steps,
        pct_start=max(0.05, a.warmup), anneal_strategy='cos')
    scaler = GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smooth)

    best = -1.0; best_probs = None
    os.makedirs(a.out, exist_ok=True)
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if a.mixup > 0 and np.random.rand() < 0.5:
                x, ya, yb, lam = mixup_cutmix(x, y, a.mixup)
                with autocast():
                    out = model(x)
                    loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
            else:
                with autocast():
                    out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0)
        probs, _ = predict_probs(model, va_dl, device, tta=False)
        acc = accuracy_score(va.label.values, probs.argmax(1))
        print(f'[{a.backbone} f{fold} s{a.seed}] ep {ep+1}/{a.epochs} loss {tot/len(tr):.4f} '
              f'val_acc {acc:.4f} {time.time()-t0:.0f}s', flush=True)
        if acc > best:
            best = acc; best_probs = probs
            torch.save(model.state_dict(), f'{a.out}/f{fold}_best.pt')
    np.save(f'{a.out}/f{fold}_val_probs.npy', best_probs)
    np.save(f'{a.out}/f{fold}_val_idx.npy', va.index.values)
    print(f'[{a.backbone} f{fold}] BEST val_acc {best:.4f}', flush=True)

    # test prediction with TTA
    if not a.no_tta or True:
        ck = f'{a.out}/f{fold}_best.pt'
        model.load_state_dict(torch.load(ck, map_location=device))
        sub = pd.read_csv(a.sample_sub)
        te_dl = DataLoader(CassavaDS(sub, a.test_images, build_tfms(a.size, False), False),
                           batch_size=a.bs, shuffle=False, **dl_kw)
        tp, ids = predict_probs(model, te_dl, device, tta=not a.no_tta)
        np.save(f'{a.out}/f{fold}_test_probs.npy', tp)
        with open(f'{a.out}/f{fold}_test_ids.json', 'w') as f:
            json.dump(ids, f)
    del model; torch.cuda.empty_cache()
    return best


def run_full(a, df, device):
    set_seed(a.seed)
    tr = df.reset_index(drop=True)
    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    tr_dl = DataLoader(CassavaDS(tr, a.images, build_tfms(a.size, True), True),
                       batch_size=a.bs, shuffle=True, drop_last=True, **dl_kw)
    model = timm.create_model(a.backbone, pretrained=True, num_classes=5, drop_path_rate=a.drop_path).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(tr_dl) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=steps, pct_start=max(0.05, a.warmup), anneal_strategy='cos')
    scaler = GradScaler(); crit = nn.CrossEntropyLoss(label_smoothing=a.label_smooth)
    os.makedirs(a.out, exist_ok=True)
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if a.mixup > 0 and np.random.rand() < 0.5:
                x, ya, yb, lam = mixup_cutmix(x, y, a.mixup)
                with autocast():
                    out = model(x); loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
            else:
                with autocast():
                    out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0)
        print(f'[FULL {a.backbone} s{a.seed}] ep {ep+1}/{a.epochs} loss {tot/len(tr):.4f} {time.time()-t0:.0f}s', flush=True)
    torch.save(model.state_dict(), f'{a.out}/full_best.pt')
    sub = pd.read_csv(a.sample_sub)
    te_dl = DataLoader(CassavaDS(sub, a.test_images, build_tfms(a.size, False), False),
                       batch_size=a.bs, shuffle=False, **dl_kw)
    tp, ids = predict_probs(model, te_dl, device, tta=not a.no_tta)
    np.save(f'{a.out}/full_test_probs.npy', tp)
    with open(f'{a.out}/full_test_ids.json', 'w') as f:
        json.dump(ids, f)
    print('FULL done, test probs saved', flush=True)


def main():
    a = parse()
    torch.backends.cudnn.benchmark = True
    device = 'cuda'
    df = pd.read_csv(a.split)
    os.makedirs(a.out, exist_ok=True)
    with open(f'{a.out}/args.json', 'w') as f:
        json.dump(vars(a), f, indent=2)
    folds = [int(x) for x in a.folds.split(',')]
    scores = {}
    if a.full:
        run_full(a, df, device)
    else:
        for f in folds:
            scores[f] = run_fold(a, f, df, device)
        print('SCORES', scores, 'mean', float(np.mean(list(scores.values()))), flush=True)
        with open(f'{a.out}/scores.json', 'w') as f:
            json.dump(scores, f)


if __name__ == '__main__':
    main()
