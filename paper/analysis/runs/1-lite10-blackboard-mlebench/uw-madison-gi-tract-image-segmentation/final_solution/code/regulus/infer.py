import os, sys, json, time, argparse
import numpy as np, pandas as pd
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/regulus_alpha')
from train import build_prev_next

CACHE = '/download_job_code/payload/sandbox/workspaces/regulus_alpha/cache'
ROOT = '/download_job_code/payload/sandbox'


class InferDS(Dataset):
    def __init__(self, gidx, prev, nxt, img):
        self.gidx = np.asarray(gidx)
        self.prev = np.asarray(prev)
        self.next = np.asarray(nxt)
        self.img = img
        self.N = len(self.gidx)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        gi = self.gidx[i]
        x = np.stack([self.img[self.prev[i]], self.img[gi], self.img[self.next[i]]], axis=-1)
        x = x.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.25
        return torch.from_numpy(x).permute(2, 0, 1).contiguous(), i


def _probs(model, x, mc):
    out = model(x)
    if mc:
        return torch.softmax(out.float(), 1)[:, 1:4]
    return torch.sigmoid(out)


@torch.no_grad()
def run(model, dl, device, amp, tta, mc):
    outs = None
    n_total = len(dl.dataset)
    S = None
    for x, idx in dl:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            p = _probs(model, x, mc)
            if 'h' in tta:
                p = p + torch.flip(_probs(model, torch.flip(x, [3]), mc), [3])
            if 'v' in tta:
                p = p + torch.flip(_probs(model, torch.flip(x, [2]), mc), [2])
        p = p / (1 + ('h' in tta) + ('v' in tta))
        p = (p.float().cpu().numpy() * 255.0).round().astype(np.uint8)
        if outs is None:
            S = p.shape[-1]
            outs = np.empty((n_total, 3, S, S), dtype=np.uint8)
        outs[idx.numpy()] = p
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', choices=['val', 'test'], required=True)
    ap.add_argument('--size', type=int, default=320)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--tta', default='hv')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    a.out = a.out if a.out.startswith('/') else os.path.join(ROOT, a.out)
    ck = torch.load(a.ckpt, map_location='cpu')
    cargs = ck['args']
    encoder = cargs['encoder']
    mc = bool(cargs.get('multiclass', 0))
    device = 'cuda'
    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3,
                     classes=(4 if mc else 3), activation=None)
    model.load_state_dict(ck['model'])
    model = model.to(device).eval()
    print('loaded', a.ckpt, 'val_dice', ck.get('val_dice'), 'encoder', encoder, flush=True)

    if a.split == 'val':
        meta = pd.read_csv(f'{CACHE}/train_meta_{a.size}.csv')
        vj = json.load(open('/download_job_code/payload/sandbox/workspaces/regulus_alpha/val_cases.json'))
        gidx = np.where(meta['case'].isin(set(vj['val_cases'])).values)[0]
        img = np.memmap(f'{CACHE}/train_img_{a.size}.dat', dtype=np.uint8, mode='r', shape=(len(meta), a.size, a.size))
        prev, nxt = build_prev_next(meta)
    else:
        meta = pd.read_csv(f'{CACHE}/test_meta_{a.size}.csv')
        gidx = np.arange(len(meta))
        img = np.memmap(f'{CACHE}/test_img_{a.size}.dat', dtype=np.uint8, mode='r', shape=(len(meta), a.size, a.size))
        prev, nxt = build_prev_next(meta)

    ds = InferDS(gidx, prev[gidx], nxt[gidx], img)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
    t0 = time.time()
    probs = run(model, dl, device, True, a.tta, mc)
    print('infer done', probs.shape, time.time() - t0, flush=True)
    np.save(a.out + '_probs.npy', probs)
    meta.iloc[gidx][['id', 'case', 'day', 'slice', 'path']].to_csv(a.out + '_ids.csv', index=False)
    print('saved', a.out, flush=True)


if __name__ == '__main__':
    main()
