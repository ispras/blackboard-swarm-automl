import os, sys, time, math, argparse, json, random
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score

cv2.setNumThreads(0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rand_bbox(size, lam):
    W = size[2]; H = size[3]
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W); y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mix_batch(x, y, crit, model, a):
    if a.mixup <= 0 and a.cutmix <= 0:
        return crit(model(x), y)
    if np.random.rand() > 1 - a.mix_p:
        return crit(model(x), y)
    perm = torch.randperm(x.size(0), device=x.device)
    use_cutmix = a.cutmix > 0 and (a.mixup <= 0 or np.random.rand() < 0.5)
    alpha = a.cutmix if use_cutmix else a.mixup
    lam = np.random.beta(alpha, alpha)
    if use_cutmix:
        x1, y1, x2, y2 = rand_bbox(x.size(), lam)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2].clone()
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    else:
        lam = max(lam, 1 - lam)
        x = lam * x + (1 - lam) * x[perm]
    out = model(x)
    return lam * crit(out, y) + (1 - lam) * crit(out, y[perm])


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', default='tf_efficientnet_b4_ns')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--split', default='workspaces/shared/folds5_seed42.csv')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--images', default='data/train_images')
    p.add_argument('--out', default='workspaces/capella_kappa/run')
    p.add_argument('--class_weights', type=int, default=0)
    p.add_argument('--label_smooth', type=float, default=0.05)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--full', type=int, default=0, help='train on all data (no val)')
    p.add_argument('--init', default='', help='init weights from checkpoint')
    p.add_argument('--mixup', type=float, default=0.0)
    p.add_argument('--cutmix', type=float, default=0.0)
    p.add_argument('--mix_p', type=float, default=0.5)
    p.add_argument('--test_pred', type=int, default=1)
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
            A.Rotate(limit=25, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=size//20, max_width=size//20, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


@torch.no_grad()
def predict(model, loader, device, tta=False):
    model.eval()
    preds, ids = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            logits = model(x)
            if tta:
                logits = logits + model(torch.flip(x, dims=[3]))
            logits = logits / (2 if tta else 1)
        preds.append(logits.float().cpu())
        ids.extend(y)
    return torch.cat(preds), ids


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    device = 'cuda'
    df = pd.read_csv(a.split)
    if a.full:
        tr = df.copy(); va = df.iloc[:0].copy()
    else:
        tr = df[df.fold != a.fold].copy(); va = df[df.fold == a.fold].copy()

    tr_ds = CassavaDS(tr, a.images, build_tfms(a.size, True), True)
    va_ds = CassavaDS(va, a.images, build_tfms(a.size, False), True)
    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    tr_dl = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, drop_last=True, **dl_kw)
    va_dl = DataLoader(va_ds, batch_size=a.bs, shuffle=False, **dl_kw)

    model = timm.create_model(a.backbone, pretrained=True, num_classes=5).to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device))
        print('initialized from', a.init, flush=True)
    if a.class_weights:
        cnt = tr.label.value_counts().sort_index().values
        w = (cnt.sum() / (5 * cnt))
        w = torch.tensor(w, dtype=torch.float32, device=device)
        print('class weights', w.tolist())
    else:
        w = None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=a.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(tr_dl) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=steps,
        pct_start=min(0.3, max(0.05, a.warmup / max(a.epochs, 2))),
        anneal_strategy='cos')
    scaler = GradScaler()

    best = -1
    log = open(os.path.join(a.out, 'train.log'), 'a')
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                loss = mix_batch(x, y, crit, model, a)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0)
        msg = f'[{a.backbone} s{a.seed}] ep {ep+1}/{a.epochs} loss {tot/len(tr_ds):.4f} time {time.time()-t0:.0f}s'
        if len(va) > 0:
            logits, ids = predict(model, va_dl, device)
            acc = accuracy_score(va.label.values, logits.argmax(1).numpy())
            msg += f' val_acc {acc:.4f}'
            if acc > best:
                best = acc
                torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
    log.write(f'BEST {best:.4f}\n'); log.close()
    torch.save(model.state_dict(), os.path.join(a.out, 'last.pt'))
    with open(os.path.join(a.out, 'best_acc.txt'), 'w') as f:
        f.write(str(best))

    if a.test_pred:
        ck = os.path.join(a.out, 'best.pt') if os.path.exists(os.path.join(a.out, 'best.pt')) else os.path.join(a.out, 'last.pt')
        model.load_state_dict(torch.load(ck, map_location=device))
        if len(va) > 0:
            vlog, _ = predict(model, va_dl, device, tta=True)
            vprobs = F.softmax(vlog, 1).numpy()
            np.save(os.path.join(a.out, 'val_probs.npy'), vprobs)
            va[['image_id']].to_csv(os.path.join(a.out, 'val_ids.csv'), index=False)
            np.save(os.path.join(a.out, 'val_idx.npy'), va.index.values)
        sub = pd.read_csv('data/sample_submission.csv')
        te_ds = CassavaDS(sub, 'data/test_images', build_tfms(a.size, False), train=False)
        te_dl = DataLoader(te_ds, batch_size=a.bs, shuffle=False, **dl_kw)
        logits, ids = predict(model, te_dl, device, tta=True)
        probs = F.softmax(logits, 1).numpy()
        np.save(os.path.join(a.out, 'test_probs.npy'), probs)
        with open(os.path.join(a.out, 'test_ids.json'), 'w') as f:
            json.dump(ids, f)
        pred = probs.argmax(1)
        out = pd.DataFrame({'image_id': ids, 'label': pred})
        out.to_csv(os.path.join(a.out, 'submission.csv'), index=False)
        print('wrote submission', out.shape, flush=True)


if __name__ == '__main__':
    main()
