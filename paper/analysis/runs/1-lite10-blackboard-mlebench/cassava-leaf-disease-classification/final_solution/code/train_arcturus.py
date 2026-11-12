import os, sys, time, argparse, json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import accuracy_score, f1_score

DATA = "/download_job_code/payload/sandbox/data"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficientnet_b4")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--smooth", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/effb4")
    p.add_argument("--folds_csv", default="folds.csv")
    p.add_argument("--allfolds", action="store_true")
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--mix_prob", type=float, default=0.5)
    p.add_argument("--train_full", action="store_true")
    return p.parse_args()


def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    return y1, y2, x1, x2


def mixed_criterion(crit, out, y, y2, lam):
    return lam * crit(out, y) + (1 - lam) * crit(out, y2)


class CassavaDS(Dataset):
    def __init__(self, df, img_dir, tfm, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm = tfm
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(os.path.join(self.img_dir, r.image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(image=img)["image"]
        if self.has_label:
            return img, int(r.label)
        return img


def build_tf(img_size, train):
    if train:
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(int(img_size * 1.14), int(img_size * 1.14)),
        A.CenterCrop(img_size, img_size),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])


def run_fold(args, fold, df):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.train_full:
        tr = df
        va = df[df.fold == 0]
    else:
        tr = df[df.fold != fold]
        va = df[df.fold == fold]
    tr_ds = CassavaDS(tr, f"{DATA}/train_images", build_tf(args.img_size, True))
    va_ds = CassavaDS(va, f"{DATA}/train_images", build_tf(args.img_size, False))
    tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    va_ld = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=5).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = len(tr_ld) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=args.smooth)

    best = 0.0
    os.makedirs(args.out, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for x, y in tr_ld:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            opt.zero_grad()
            use_mix = (args.mixup > 0 or args.cutmix > 0) and np.random.rand() < args.mix_prob
            with torch.cuda.amp.autocast():
                if use_mix:
                    perm = torch.randperm(x.size(0), device=x.device)
                    y2 = y[perm]
                    if args.cutmix > 0 and (args.mixup <= 0 or np.random.rand() < 0.5):
                        lam = np.random.beta(args.cutmix, args.cutmix)
                        y1b, y2b, x1b, x2b = rand_bbox(x.size(2), x.size(3), lam)
                        x[:, :, y1b:y2b, x1b:x2b] = x[perm, :, y1b:y2b, x1b:x2b]
                        lam = 1 - ((y2b - y1b) * (x2b - x1b) / (x.size(2) * x.size(3)))
                    else:
                        lam = np.random.beta(args.mixup, args.mixup)
                        x = lam * x + (1 - lam) * x[perm]
                    out = model(x)
                    loss = mixed_criterion(crit, out, y, y2, lam)
                else:
                    out = model(x)
                    loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * x.size(0)
        # val
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for x, y in va_ld:
                x = x.cuda(non_blocking=True)
                with torch.cuda.amp.autocast():
                    out = model(x)
                probs.append(F.softmax(out.float(), 1).cpu().numpy())
                ys.append(y.numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        acc = accuracy_score(ys, probs.argmax(1))
        f1 = f1_score(ys, probs.argmax(1), average="macro")
        print(f"[f{fold}] ep{ep+1}/{args.epochs} loss {tot/len(tr):.4f} "
              f"val_acc {acc:.4f} f1 {f1:.4f} {time.time()-t0:.0f}s", flush=True)
        if acc > best or (args.train_full and ep == args.epochs - 1):
            best = max(best, acc)
            torch.save(model.state_dict(), f"{args.out}/f{fold}_best.pt")
            np.save(f"{args.out}/f{fold}_val_probs.npy", probs)
            np.save(f"{args.out}/f{fold}_val_idx.npy", va.index.values)
    print(f"[f{fold}] BEST val_acc {best:.4f}", flush=True)
    return best


def main():
    args = parse()
    df = pd.read_csv(args.folds_csv)
    torch.backends.cudnn.benchmark = True
    if args.allfolds:
        scores = []
        for f in range(args.folds):
            scores.append(run_fold(args, f, df))
        print("ALL FOLDS", scores, "mean", np.mean(scores), flush=True)
    else:
        run_fold(args, args.fold, df)


if __name__ == "__main__":
    main()
