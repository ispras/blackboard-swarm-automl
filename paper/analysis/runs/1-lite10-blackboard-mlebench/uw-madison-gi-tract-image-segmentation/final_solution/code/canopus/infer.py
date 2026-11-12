import os, sys, json, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import GISegDataset, GITestDataset, CACHE_TR, CACHE_TE

ROOT = "/download_job_code/payload/sandbox"
WS = f"{ROOT}/workspaces/canopus_phi"

def load_model(ckpt):
    ck = torch.load(ckpt, map_location='cpu')
    a = ck['args']
    m = getattr(smp, a['arch'])(encoder_name=a['encoder'], encoder_weights=None,
                                in_channels=a.get('stack', 3), classes=3)
    m.load_state_dict(ck['state']); m.cuda().eval()
    return m, a

@torch.no_grad()
def predict_probs(model, ds, out_path, bs=32, workers=6, tta=True):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    n = len(ds)
    mm = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.uint8, shape=(n, 3, 320, 320))
    k = 0
    for x in dl:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = p + torch.sigmoid(model(torch.flip(x, [2])).float()).flip([2])
                p = p + torch.sigmoid(model(torch.flip(x, [3])).float()).flip([3])
                p = p + torch.sigmoid(model(torch.flip(x, [2, 3])).float()).flip([2, 3])
                p = p / 4
        p = (p.clamp(0, 1) * 255).round().byte().cpu().numpy()
        mm[k:k+len(p)] = p; k += len(p)
    mm.flush()
    print('wrote', out_path, n, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='val', choices=['val','test','train'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--no_tta', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    model, a = load_model(args.ckpt); stack = a.get('stack', 3)
    if args.split == 'test':
        index = pd.read_csv(f'{CACHE_TE}/index.csv')
        images = np.load(f'{CACHE_TE}/images_u8.npy', mmap_mode='r')
        ds = GITestDataset(index, images, stack)
    else:
        index = pd.read_csv(f'{CACHE_TR}/index.csv')
        images = np.load(f'{CACHE_TR}/images_u8.npy', mmap_mode='r')
        masks = np.load(f'{CACHE_TR}/masks_u8.npy', mmap_mode='r')
        if args.split == 'val':
            vc = set(json.load(open(f'{ROOT}/workspaces/regulus_alpha/val_cases.json'))['val_cases'])
            idx = np.where(index['case'].isin(vc).values)[0]
        else:
            idx = np.arange(len(index))
        if args.limit: idx = idx[:args.limit]
        ds = GISegDataset(index, images, masks, False, idx, stack)
    predict_probs(model, ds, args.out, args.bs, args.workers, not args.no_tta)
    if args.split == 'test':
        ids = index['id'].values
    else:
        ids = index.iloc[idx]['id'].values
    pd.DataFrame({'id': ids}).to_csv(args.out.replace('.npy','_ids.csv'), index=False)
    print('wrote ids', len(ids), flush=True)

if __name__ == '__main__':
    main()
