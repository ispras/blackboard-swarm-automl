import argparse, os, time, json, random
import numpy as np, pandas as pd
from PIL import Image
import cv2
cv2.setNumThreads(2)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA = '/download_job_code/payload/sandbox/data'
ROOT = '/download_job_code/payload/sandbox'


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class CassavaDS(Dataset):
    def __init__(self, df, img_dir, img_size, train):
        self.df = df.reset_index(drop=True)
        self.dir = img_dir
        if train:
            self.tf = A.Compose([
                A.RandomResizedCrop(img_size, img_size, scale=(0.6, 1.0), ratio=(0.75, 1.333), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05, p=0.7),
                A.CoarseDropout(max_holes=8, max_height=img_size // 12, max_width=img_size // 12, p=0.35),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(os.path.join(self.dir, r.image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tf(image=img)['image']
        return img, int(r.label)


def evaluate(model, loader, device):
    model.eval()
    n = 0; correct = 0; probs_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device)
            with torch.cuda.amp.autocast():
                logits = model(x)
            p = torch.softmax(logits.float(), 1)
            probs_all.append(p.cpu().numpy())
            correct += (p.argmax(1) == y).sum().item(); n += y.numel()
    return correct / n, np.concatenate(probs_all)


def predict_tta(model, loader, device, img_size):
    model.eval(); probs_all = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            acc = None
            for flip in (False, True):
                xx = torch.flip(x, [3]) if flip else x
                with torch.cuda.amp.autocast():
                    logits = model(xx)
                p = torch.softmax(logits.float(), 1)
                acc = p if acc is None else acc + p
            probs_all.append((acc / 2).cpu().numpy())
    return np.concatenate(probs_all)


class Mixup:
    def __init__(self, alpha=0.8, prob=0.5, num_classes=5, smooth=0.1):
        self.alpha = alpha; self.prob = prob; self.nc = num_classes; self.smooth = smooth

    def __call__(self, x, y):
        y1 = torch.zeros(y.size(0), self.nc, device=y.device)
        y1.scatter_(1, y[:, None], 1.0)
        y1 = y1 * (1 - self.smooth) + self.smooth / self.nc
        if random.random() > self.prob:
            return x, y1
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        y1 = lam * y1 + (1 - lam) * y1[idx]
        return x, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='convnext_base')
    ap.add_argument('--img-size', type=int, default=384)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--min-lr', type=float, default=1e-5)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out', default='workspaces/bellatrix_delta/run')
    ap.add_argument('--mixup-prob', type=float, default=0.5)
    ap.add_argument('--smooth', type=float, default=0.1)
    ap.add_argument('--drop-path', type=float, default=0.1)
    ap.add_argument('--extra-train-on-val', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    split = pd.read_csv(os.path.join(ROOT, 'workspaces/aldebaran_epsilon/cv_split.csv'))
    tr = split[split.fold != args.fold].reset_index(drop=True)
    va = split[split.fold == args.fold].reset_index(drop=True)
    va_ids = va.copy()
    if args.limit:
        tr = tr.iloc[:args.limit].reset_index(drop=True); va = va.iloc[:args.limit].reset_index(drop=True); va_ids = va.copy()
    print(f'train={len(tr)} val={len(va)}', flush=True)

    tr_ds = CassavaDS(tr, os.path.join(DATA, 'train_images'), args.img_size, True)
    va_ds = CassavaDS(va, os.path.join(DATA, 'train_images'), args.img_size, False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=2)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=False)

    device = 'cuda'
    model = timm.create_model(args.model, pretrained=True, num_classes=5, drop_path_rate=args.drop_path)
    model = model.to(device)

    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        if p_.ndim == 1 or n_.endswith('.bias'):
            no_decay.append(p_)
        else:
            decay.append(p_)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.wd},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr)
    steps = len(tr_ld) * args.epochs
    warm = len(tr_ld) * args.warmup

    def lr_lambda(step):
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, steps - warm)
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()
    mix = Mixup(prob=args.mixup_prob, smooth=args.smooth)
    ce = nn.CrossEntropyLoss()

    best = 0.0; step = 0
    log = open(os.path.join(args.out, 'train.log'), 'w')
    for ep in range(args.epochs):
        model.train(); tl = 0.0; t0 = time.time()
        for x, y in tr_ld:
            x = x.to(device, non_blocking=True); y = y.to(device)
            xm, ym = mix(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(xm)
                loss = torch.sum(-ym * torch.log_softmax(out.float(), 1), 1).mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            tl += loss.item()
        acc, _ = evaluate(model, va_ld, device)
        msg = f'epoch {ep+1}/{args.epochs} loss {tl/len(tr_ld):.4f} val_acc {acc:.4f} lr {sched.get_last_lr()[0]:.2e} time {time.time()-t0:.0f}s'
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
        if acc > best:
            best = acc
            torch.save({'model': model.state_dict(), 'acc': acc, 'args': vars(args)},
                       os.path.join(args.out, 'best.pt'))
            print(f'  saved best {best:.4f}', flush=True)
    log.close()

    ck = torch.load(os.path.join(args.out, 'best.pt'), map_location=device)
    model.load_state_dict(ck['model'])
    val_acc, val_probs = evaluate(model, va_ld, device)
    te = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))[['image_id']]
    te_ds = CassavaDS(te.rename(columns={'image_id': 'image_id'}).assign(label=0),
                      os.path.join(DATA, 'test_images'), args.img_size, False)
    te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_probs = predict_tta(model, va_ld, device, args.img_size)
    test_probs = predict_tta(model, te_ld, device, args.img_size)
    np.save(os.path.join(args.out, 'val_probs_fold0.npy'), val_probs)
    np.save(os.path.join(args.out, 'test_probs.npy'), test_probs)
    va_ids.to_csv(os.path.join(args.out, 'val_ids_fold0.csv'), index=False)
    pred = test_probs.argmax(1)
    sub = pd.DataFrame({'image_id': te.image_id, 'label': pred})
    sub.to_csv(os.path.join(args.out, 'submission.csv'), index=False)
    tta_acc = (val_probs.argmax(1) == va.label.values).mean()
    with open(os.path.join(args.out, 'val_acc.txt'), 'w') as f:
        f.write(f'best_val_acc={best:.5f}\ntta_val_acc={tta_acc:.5f}\n')
    print(f'DONE best={best:.5f} tta={tta_acc:.5f}', flush=True)


if __name__ == '__main__':
    main()
