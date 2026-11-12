import argparse, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Cassava, seed_all
import timm


def infer(model, ld, nflip):
    model.eval(); out = []
    with torch.no_grad():
        for x, y in ld:
            x = x.to('cuda', non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                acc = None
                flips = [(False, False)]
                if nflip == 2:
                    flips = [(False, False), (True, False)]
                elif nflip == 4:
                    flips = [(False, False), (True, False), (False, True), (True, True)]
                for fh, fv in flips:
                    xi = x
                    if fh: xi = torch.flip(xi, dims=[3])
                    if fv: xi = torch.flip(xi, dims=[2])
                    p = torch.softmax(model(xi).float(), 1)
                    acc = p if acc is None else acc + p
                p = acc / len(flips)
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sizes', default='352,384,416,448')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    seed_all(42)
    root = '/download_job_code/payload/sandbox'
    spl = pd.read_csv(os.path.join(root, 'workspaces/aldebaran_epsilon/cv_split.csv'))
    va = spl[spl.fold == 0].reset_index(drop=True); y = va.label.values
    state = torch.load(args.ckpt)
    for s in [int(x) for x in args.sizes.split(',')]:
        try:
            model = timm.create_model(args.model, pretrained=False, num_classes=5, img_size=s).to('cuda')
        except TypeError:
            model = timm.create_model(args.model, pretrained=False, num_classes=5).to('cuda')
        r = model.load_state_dict(state, strict=False)
        if r.missing_keys or r.unexpected_keys:
            print('load', s, 'missing', r.missing_keys[:3], 'unexpected', r.unexpected_keys[:3], flush=True)
        model.eval()
        ds = Cassava(va, os.path.join(root, 'data/train_images'), s, False)
        ld = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
        for nf in (2, 4):
            p = infer(model, ld, nf)
            print(f'size {s} flips {nf} val_acc {(p.argmax(1)==y).mean():.5f}', flush=True)
            np.save(f'/tmp/opencode/{args.model}_{s}_{nf}.npy', p)


if __name__ == '__main__':
    main()
