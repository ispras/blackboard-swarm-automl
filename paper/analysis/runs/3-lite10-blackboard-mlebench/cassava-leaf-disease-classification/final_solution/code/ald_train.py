import argparse, os, random, time, json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


class Cassava(Dataset):
    def __init__(self, df, img_dir, size, train=True):
        self.df = df.reset_index(drop=True); self.img_dir = img_dir; self.train = train
        if train:
            self.tf = A.Compose([
                A.RandomResizedCrop(height=size, width=size, scale=(0.65, 1.0), ratio=(0.8, 1.25), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.6),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.Resize(height=size, width=size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(os.path.join(self.img_dir, r.image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tf(image=img)['image']
        if 'label' in self.df.columns:
            return img, int(r.label)
        return img, r.image_id


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix(x, y):
    r = np.random.rand()
    if r < 0.3:  # mixup
        lam = np.random.beta(0.4, 0.4)
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        return x, y, y[idx], lam
    elif r < 0.5:  # cutmix
        lam = np.random.beta(1.0, 1.0)
        idx = torch.randperm(x.size(0), device=x.device)
        x1, y1, x2, y2 = rand_bbox(x.size(), lam)
        x[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
        return x, y, y[idx], lam
    return x, y, y, 1.0


def run_eval(model, loader, device, tta=False):
    model.eval(); probs = []; ids = []; ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                if tta:
                    logits = logits + model(torch.flip(x, dims=[3]))
                p = torch.softmax(logits.float(), 1)
            probs.append(p.cpu().numpy())
            if isinstance(y, torch.Tensor):
                ys.append(y.numpy())
            else:
                ids.extend(list(y))
    return np.concatenate(probs), (np.concatenate(ys) if ys else None), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='swin_base_patch4_window12_384')
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--tag', default='swin')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    seed_all(42)
    dev = 'cuda'; ncls = 5
    root = '/download_job_code/payload/sandbox'
    out = os.path.join(root, 'workspaces/aldebaran_epsilon', args.tag)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, 'args.json'), 'w'))

    spl = pd.read_csv(os.path.join(root, 'workspaces/aldebaran_epsilon/cv_split.csv'))
    tr = spl[spl.fold != 0].reset_index(drop=True)
    va = spl[spl.fold == 0].reset_index(drop=True)
    if args.smoke:
        tr = tr.sample(200, random_state=0).reset_index(drop=True)
        va = va.sample(100, random_state=0).reset_index(drop=True)
    print('train', len(tr), 'val', len(va), flush=True)

    tr_ds = Cassava(tr, os.path.join(root, 'data/train_images'), args.size, True)
    va_ds = Cassava(va, os.path.join(root, 'data/train_images'), args.size, False)
    tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    va_ld = DataLoader(va_ds, batch_size=args.bs * 2, shuffle=False, num_workers=args.workers,
                       pin_memory=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=ncls).to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = len(tr_ld)
    total = args.epochs * steps
    warm = args.warmup * steps
    def lr_at(step):
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.cuda.amp.GradScaler()

    best = -1; gstep = 0; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); losses = []
        for x, y in tr_ld:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            if np.random.rand() < 0.5:
                x, ya, yb, lam = mixup_cutmix(x, y)
                logits = model(x)
                loss = crit(logits, ya) * lam + crit(logits, yb) * (1 - lam)
            else:
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item()); gstep += 1
            if gstep % 50 == 0:
                print(f'  step {gstep}/{total} loss {np.mean(losses[-50:]):.4f} t {time.time()-t0:.0f}s', flush=True)
        p, yy, _ = run_eval(model, va_ld, dev)
        acc = (p.argmax(1) == yy).mean()
        print(f'ep {ep+1}/{args.epochs} loss {np.mean(losses):.4f} val_acc {acc:.4f} lr {opt.param_groups[0]["lr"]:.2e} t {time.time()-t0:.0f}s', flush=True)
        if acc > best:
            best = acc; torch.save(model.state_dict(), os.path.join(out, 'best.pt'))
    model.load_state_dict(torch.load(os.path.join(out, 'best.pt')))
    # val probs with hflip TTA
    p, yy, _ = run_eval(model, va_ld, dev, tta=True)
    acc = (p.argmax(1) == yy).mean()
    np.save(os.path.join(out, 'val_probs.npy'), p)
    va[['image_id']].to_csv(os.path.join(out, 'val_ids.csv'), index=False)
    print(f'FINAL fold0 acc {acc:.4f} best_during {best:.4f}', flush=True)

    # test probs
    ss = pd.read_csv(os.path.join(root, 'data/sample_submission.csv'))
    te_ds = Cassava(ss[['image_id']], os.path.join(root, 'data/test_images'), args.size, False)
    te_ld = DataLoader(te_ds, batch_size=args.bs * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    p, _, ids = run_eval(model, te_ld, dev, tta=True)
    np.save(os.path.join(out, 'test_probs.npy'), p)
    pd.DataFrame({'image_id': ids}).to_csv(os.path.join(out, 'test_ids.csv'), index=False)
    meta = dict(args=vars(args), fold0_acc=float(acc), best=float(best))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=2)
    print('DONE', json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
