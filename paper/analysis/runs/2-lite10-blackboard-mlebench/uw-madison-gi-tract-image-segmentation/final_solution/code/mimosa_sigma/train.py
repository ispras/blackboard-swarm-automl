import os, sys, json, time, argparse, math
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

ROOT = '/download_job_code/payload/sandbox'
C = f'{ROOT}/workspaces/mimosa_sigma/cache'
cv2.setNumThreads(1)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', default='Unet')
    p.add_argument('--encoder', default='timm-efficientnet-b4')
    p.add_argument('--encoder-weights', default='imagenet')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--epochs', type=int, default=22)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--size', type=int, default=320)
    p.add_argument('--nc', type=int, default=5)   # 2.5D channels (odd)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tag', default='b4_unet_5ch')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val-every', type=int, default=1)
    p.add_argument('--all-data', type=int, default=0)
    p.add_argument('--init-ckpt', default='')
    return p.parse_args()

class GIDataset(Dataset):
    def __init__(self, split, idxs, fold=None, train=False, nc=1, size=320):
        self.split = split; self.idxs = np.asarray(idxs); self.train = train
        self.nc = nc; self.size = size
        self.img = np.load(f'{C}/{split}_img.npy', mmap_mode='r')
        self.prev = np.load(f'{C}/{split}_prev.npy')
        self.next = np.load(f'{C}/{split}_next.npy')
        self.mask = np.load(f'{C}/train_mask.npy', mmap_mode='r') if split == 'train' else None
        h = nc // 2
        # channel index tables
        self.p_tab = []
        self.n_tab = []
        for _ in range(h):
            pass
        pcur = np.arange(len(self.prev)); ncur = np.arange(len(self.next))
        self.ps = [pcur]
        for _ in range(h):
            pcur = self.prev[pcur]; self.ps.append(pcur)
        self.ns = [ncur]
        for _ in range(h):
            ncur = self.next[ncur]; self.ns.append(ncur)
        self.hflip = A.HorizontalFlip(p=0.5)
        self.aug = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.12, rotate_limit=15,
                               border_mode=cv2.BORDER_CONSTANT, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ])

    def __len__(self):
        return len(self.idxs)

    def _stack(self, i):
        chans = [self.img[self.ps[k][i]] for k in range(len(self.ps) - 1, 0, -1)]
        chans.append(self.img[i])
        chans += [self.img[self.ns[k][i]] for k in range(1, len(self.ns))]
        return np.stack(chans, -1)  # H,W,nc

    def __getitem__(self, k):
        i = int(self.idxs[k])
        x = self._stack(i)
        if self.train:
            m = np.transpose(self.mask[i], (1, 2, 0))  # 320,320,3
            r = self.hflip(image=x, mask=m)
            x, m = r['image'], r['mask']
            r = self.aug(image=x, mask=m)
            x, m = r['image'], r['mask']
            y = np.transpose(m, (2, 0, 1)).astype(np.float32)
        else:
            y = (self.mask[i].astype(np.float32) if self.mask is not None
                 else np.zeros((3, self.size, self.size), dtype=np.float32))
        x = (x.astype(np.float32) / 255.0 - 0.5) / 0.25
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), i

def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(2, 3)) + 1.0
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1.0
    return (1 - num / den).mean()

@torch.no_grad()
def validate(model, loader, dev, thrs=(0.2, 0.3, 0.4, 0.5)):
    model.eval()
    dsum = {t: np.zeros(3) for t in thrs}; dcnt = {t: 0 for t in thrs}
    psum = {t: np.zeros(3) for t in thrs}; pcnt = {t: 0 for t in thrs}
    all_probs = []; all_ids = []
    for x, y, ids in loader:
        x = x.to(dev, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(x)
        p = torch.sigmoid(logits).float().cpu().numpy()
        yy = y.numpy()
        for t in thrs:
            pred = (p > t).astype(np.float32)
            inter = (pred * yy).sum(axis=(2, 3)); ps = pred.sum(axis=(2, 3)); gs = yy.sum(axis=(2, 3))
            d = np.where(ps + gs == 0, 1.0, 2.0 * inter / np.maximum(ps + gs, 1))
            dsum[t] += d.sum(0); dcnt[t] += d.shape[0]
            pos = gs > 0
            dpos = np.where(pos, np.where(ps + gs == 0, 1.0, 2.0 * inter / np.maximum(ps + gs, 1)), 0.0)
            psum[t] += dpos.sum(0); pcnt[t] += pos.sum(0)
        all_probs.append(p.astype(np.float16)); all_ids.append(ids.numpy())
    dall = {t: dsum[t] / dcnt[t] for t in thrs}
    dpos = {t: psum[t] / np.maximum(pcnt[t], 1) for t in thrs}
    best_t = max(thrs, key=lambda t: dall[t].mean())
    return dall[best_t], dpos[best_t], dall, best_t, np.concatenate(all_probs), np.concatenate(all_ids)

def main():
    a = get_args()
    os.makedirs(f'{ROOT}/workspaces/mimosa_sigma/models', exist_ok=True)
    os.makedirs(f'{ROOT}/workspaces/mimosa_sigma/logs', exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    ids = json.load(open(f'{C}/train_ids.json'))
    fold = np.load(f'{C}/train_fold.npy')
    tr_idx = np.where(fold != a.fold)[0]
    va_idx = np.where(fold == a.fold)[0]
    if a.all_data:
        tr_idx = np.arange(len(ids))
        print('ALL-DATA mode: val fold is included in training (metrics optimistic)', flush=True)
    print(f'train {len(tr_idx)} val {len(va_idx)} nc {a.nc}', flush=True)
    tr_ds = GIDataset('train', tr_idx, train=True, nc=a.nc, size=a.size)
    va_ds = GIDataset('train', va_idx, train=False, nc=a.nc, size=a.size)
    tr_dl = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    va_dl = DataLoader(va_ds, batch_size=a.bs, shuffle=False, num_workers=a.workers,
                       pin_memory=True, persistent_workers=True)
    kw = dict(encoder_name=a.encoder, encoder_weights=a.encoder_weights,
              in_channels=a.nc, classes=3)
    model = getattr(smp, a.arch)(**kw).to(dev)
    if a.init_ckpt:
        ck0 = torch.load(a.init_ckpt, map_location=dev)
        model.load_state_dict(ck0['model'])
        print('initialized from', a.init_ckpt, 'ep', ck0.get('ep'), flush=True)
    if 'mit' in a.encoder:
        pass
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.02)
    scaler = torch.cuda.amp.GradScaler()
    bce = nn.BCEWithLogitsLoss()
    best = -1
    logf = open(f'{ROOT}/workspaces/mimosa_sigma/logs/{a.tag}_f{a.fold}.log', 'a')
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for x, y, _ in tr_dl:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = bce(logits, y) + dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            run += loss.item(); nb += 1
        sched.step()
        msg = f'ep {ep} loss {run/nb:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s'
        if ep % a.val_every == 0 or ep == a.epochs:
            dice, dpos, dall, bt, probs, vids = validate(model, va_dl, dev)
            score = float(dice.mean())
            msg += f' dice@{bt} ' + ' '.join(f'{d:.4f}' for d in dice) + f' mean {score:.4f}'
            msg += ' pos ' + ' '.join(f'{d:.4f}' for d in dpos) + f' posmean {dpos.mean():.4f}'
            torch.save({'model': model.state_dict(), 'dice': dice.tolist(), 'ep': ep,
                        'best_thr': bt, 'args': vars(a)},
                       f'{ROOT}/workspaces/mimosa_sigma/models/{a.tag}_f{a.fold}_last.pt')
            if score > best:
                best = score
                torch.save({'model': model.state_dict(), 'dice': dice.tolist(), 'ep': ep,
                            'best_thr': bt, 'args': vars(a)}, f'{ROOT}/workspaces/mimosa_sigma/models/{a.tag}_f{a.fold}.pt')
                np.save(f'{ROOT}/workspaces/mimosa_sigma/models/{a.tag}_f{a.fold}_valprobs.npy', probs)
                np.save(f'{ROOT}/workspaces/mimosa_sigma/models/{a.tag}_f{a.fold}_valid.npy', vids)
                msg += ' *best'
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()
    # final: load best and predict test with hflip TTA
    ckpath = (f'{ROOT}/workspaces/mimosa_sigma/models/{a.tag}_f{a.fold}_last.pt' if a.all_data
              else f'{ROOT}/workspaces/mimosa_sigma/models/{a.tag}_f{a.fold}.pt')
    ck = torch.load(ckpath, map_location=dev)
    model.load_state_dict(ck['model']); model.eval()
    te_ds = GIDataset('test', np.arange(len(json.load(open(f'{C}/test_ids.json')))), train=False, nc=a.nc, size=a.size)
    te_dl = DataLoader(te_ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
    tp = []
    with torch.no_grad():
        for x, _, _ in te_dl:
            x = x.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast():
                p = torch.sigmoid(model(x)).float()
                p2 = torch.sigmoid(model(torch.flip(x, dims=[3]))).float()
            p = (p + torch.flip(p2, dims=[3])) * 0.5
            tp.append(p.cpu().numpy().astype(np.float16))
    np.save(f'{ROOT}/workspaces/mimosa_sigma/models/{a.tag}_f{a.fold}_testprobs.npy', np.concatenate(tp))
    print('saved test probs', flush=True)

if __name__ == '__main__':
    main()
