import os, sys, json, math, time, argparse
import numpy as np, pandas as pd
import cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp

cv2.setNumThreads(1)
CACHE = '/download_job_code/payload/sandbox/workspaces/regulus_alpha/cache'
VAL_JSON = '/download_job_code/payload/sandbox/workspaces/regulus_alpha/val_cases.json'


def get_aug(size, mild=False):
    if mild:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03),
                     rotate=(-10, 10), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        ])
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.20), translate_percent=(-0.0625, 0.0625),
                 rotate=(-25, 25), border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.OneOf([A.ElasticTransform(alpha=1, sigma=20, p=1.0),
                 A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0)], p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    ])


def build_prev_next(meta):
    day_n = meta['day_n'].values
    pos = meta['pos'].values
    ar = np.arange(len(meta))
    prev = np.where(pos > 0, ar - 1, ar)
    nxt = np.where(pos < day_n - 1, ar + 1, ar)
    return prev, nxt


class GISeg(Dataset):
    def __init__(self, gidx, prev, nxt, img, msk, train=False, size=320, multiclass=True, mild=False):
        self.multiclass = multiclass
        self.gidx = np.asarray(gidx)
        self.prev = np.asarray(prev)
        self.next = np.asarray(nxt)
        self.img = img
        self.msk = msk
        self.train = train
        self.aug = get_aug(size, mild) if train else None
        self.N = len(self.gidx)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        gi = self.gidx[i]
        x = np.stack([self.img[self.prev[i]], self.img[gi], self.img[self.next[i]]], axis=-1)
        m = self.msk[gi].transpose(1, 2, 0).astype(np.float32)
        if self.train:
            r = self.aug(image=x, mask=m)
            x, m = r['image'], r['mask']
        x = x.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.25
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        if self.multiclass:
            lab = np.zeros(m.shape[:2], dtype=np.int64)
            for c in range(3):
                lab[m[..., c] > 0.5] = c + 1
            return x, torch.from_numpy(lab)
        m = torch.from_numpy(m.astype(np.float32)).permute(2, 0, 1).contiguous()
        return x, m


def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum((0, 2, 3)) + eps
    den = prob.sum((0, 2, 3)) + target.sum((0, 2, 3)) + eps
    return 1 - (num / den).mean()


def batch_dice(pred, gt):
    # pred,gt: B,C,H,W bool ; returns per (b,c) dice with empty-empty=1
    inter = (pred & gt).sum((2, 3)).float()
    sp = pred.sum((2, 3)).float()
    sg = gt.sum((2, 3)).float()
    d = 2 * inter / (sp + sg + 1e-6)
    both_empty = (sp == 0) & (sg == 0)
    d = torch.where(both_empty, torch.ones_like(d), d)
    only_pred = (sg == 0) & (sp > 0)
    d = torch.where(only_pred, torch.zeros_like(d), d)
    return d


@torch.no_grad()
def validate(model, dl, device, amp, multiclass=True):
    model.eval()
    ds = []
    ds_pos = []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x)
        if multiclass:
            pr = out.argmax(1)  # B,H,W
            p = torch.stack([pr == (c + 1) for c in range(3)], 1)
            g = torch.stack([y == (c + 1) for c in range(3)], 1)
        else:
            p = torch.sigmoid(out) > 0.5
            g = y > 0.5
        d = batch_dice(p, g)
        ds.append(d.cpu())
        has = (g.sum((2, 3)) > 0)
        ds_pos.append(d[has.cpu()])
    d = torch.cat(ds)
    dp = torch.cat(ds_pos) if any(len(z) for z in ds_pos) else torch.zeros(1)
    return d.mean().item(), dp.mean().item(), d.mean(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', default='mit_b2')
    ap.add_argument('--size', type=int, default=320)
    ap.add_argument('--epochs', type=int, default=32)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default='workspaces/regulus_alpha/runs/b2')
    ap.add_argument('--val_json', default=VAL_JSON)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--all_data', type=int, default=0)
    ap.add_argument('--multiclass', type=int, default=1)
    ap.add_argument('--init', default='')
    ap.add_argument('--mild_aug', type=int, default=0)
    a = ap.parse_args()
    root = '/download_job_code/payload/sandbox'
    a.out = os.path.join(root, a.out) if not a.out.startswith('/') else a.out
    os.makedirs(a.out, exist_ok=True)
    device = 'cuda'
    torch.backends.cudnn.benchmark = True

    meta = pd.read_csv(f'{CACHE}/train_meta_{a.size}.csv')
    img = np.memmap(f'{CACHE}/train_img_{a.size}.dat', dtype=np.uint8, mode='r',
                    shape=(len(meta), a.size, a.size))
    msk = np.memmap(f'{CACHE}/train_msk_{a.size}.dat', dtype=np.uint8, mode='r',
                    shape=(len(meta), 3, a.size, a.size))
    vj = json.load(open(a.val_json))
    val_cases = set(vj['val_cases'])
    prev, nxt = build_prev_next(meta)
    is_val = meta['case'].isin(val_cases).values
    if a.all_data:
        tr_idx = np.arange(len(meta))
        va_idx = np.where(meta['case'].isin(sorted(val_cases)[:3]).values)[0]
    else:
        tr_idx = np.where(~is_val)[0]
        va_idx = np.where(is_val)[0]
    mc = bool(a.multiclass)
    tr_ds = GISeg(tr_idx, prev[tr_idx], nxt[tr_idx], img, msk, train=True, size=a.size, multiclass=mc, mild=bool(a.mild_aug))
    va_ds = GISeg(va_idx, prev[va_idx], nxt[va_idx], img, msk, train=False, size=a.size, multiclass=mc)
    print('train', len(tr_ds), 'val', len(va_ds), flush=True)

    dl_tr = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(va_ds, batch_size=a.bs * 2, shuffle=False, num_workers=a.workers,
                       pin_memory=True, persistent_workers=a.workers > 0)

    ncls = 4 if mc else 3
    model = smp.Unet(encoder_name=a.encoder, encoder_weights='imagenet',
                     in_channels=3, classes=ncls, activation=None)
    if a.init:
        ic = torch.load(a.init, map_location='cpu')
        model.load_state_dict(ic['model'])
        print('INIT from', a.init, 'prev_val', ic.get('val_dice'), flush=True)
    model = model.to(device)
    print('params(M)', sum(p.numel() for p in model.parameters()) / 1e6, flush=True)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    dice_ml = smp.losses.DiceLoss(mode='multiclass')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=a.amp == 1)
    steps_per_epoch = len(dl_tr)
    total = steps_per_epoch * a.epochs
    warmup = max(30, steps_per_epoch // 8)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    logf = open(os.path.join(a.out, 'log.txt'), 'a')
    best = -1
    gstep = 0
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        n = 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=a.amp == 1):
                out = model(x)
                if mc:
                    loss = 0.5 * ce(out, y) + 0.5 * dice_ml(out, y)
                else:
                    loss = 0.5 * bce(out, y) + 0.5 * dice_loss(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            run += loss.item(); n += 1; gstep += 1
            if a.max_steps and gstep >= a.max_steps:
                break
        tr_loss = run / max(1, n)
        if a.max_steps and gstep >= a.max_steps:
            print('smoke done', tr_loss, flush=True)
            break
        if (ep + 1) % 1 == 0:
            vd, vdp, vdc = validate(model, dl_va, device, a.amp == 1, mc)
            msg = (f'ep {ep+1}/{a.epochs} loss {tr_loss:.4f} val_dice {vd:.4f} '
                   f'val_pos {vdp:.4f} per_cls {[round(z,3) for z in vdc]} '
                   f'lr {sched.get_last_lr()[0]:.2e} t {time.time()-t0:.0f}s')
            print(msg, flush=True)
            logf.write(msg + '\n'); logf.flush()
            if vd > best:
                best = vd
                torch.save({'model': model.state_dict(), 'args': vars(a), 'val_dice': vd, 'epoch': ep + 1},
                           os.path.join(a.out, 'best.pt'))
        else:
            logf.write(f'ep {ep+1} loss {tr_loss:.4f}\n'); logf.flush()
    torch.save({'model': model.state_dict(), 'args': vars(a), 'val_dice': best, 'epoch': a.epochs},
               os.path.join(a.out, 'last.pt'))
    print('BEST', best, flush=True)
    logf.write(f'BEST {best}\n'); logf.close()


if __name__ == '__main__':
    main()
