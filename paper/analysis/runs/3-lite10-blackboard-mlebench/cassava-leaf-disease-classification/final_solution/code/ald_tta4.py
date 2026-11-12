import argparse, os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Cassava, seed_all
import timm


def infer(model, loader, do_ids=False):
    model.eval(); probs = []; ys = []; ids = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to('cuda', non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                acc = None
                for fh in (False, True):
                    for fv in (False, True):
                        xi = x
                        if fh: xi = torch.flip(xi, dims=[3])
                        if fv: xi = torch.flip(xi, dims=[2])
                        p = torch.softmax(model(xi).float(), 1)
                        acc = p if acc is None else acc + p
                p = acc / 4.0
            probs.append(p.cpu().numpy())
            if isinstance(y, torch.Tensor):
                ys.append(y.numpy())
            else:
                ids.extend(list(y))
    return np.concatenate(probs), (np.concatenate(ys) if ys else None), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val', action='store_true')
    args = ap.parse_args()
    seed_all(42)
    root = '/download_job_code/payload/sandbox'
    os.makedirs(args.out, exist_ok=True)
    model = timm.create_model(args.model, pretrained=False, num_classes=5).to('cuda')
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    if args.val:
        spl = pd.read_csv(os.path.join(root, 'workspaces/aldebaran_epsilon/cv_split.csv'))
        va = spl[spl.fold == 0].reset_index(drop=True)
        ds = Cassava(va, os.path.join(root, 'data/train_images'), args.size, False)
        ld = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
        p, yy, _ = infer(model, ld)
        acc = (p.argmax(1) == yy).mean()
        np.save(os.path.join(args.out, 'val_probs_tta4.npy'), p)
        va[['image_id']].to_csv(os.path.join(args.out, 'val_ids.csv'), index=False)
        print(f'VAL tta4 acc {acc:.5f} -> {args.out}', flush=True)
        json.dump({'val_tta4_acc': float(acc)}, open(os.path.join(args.out, 'acc_tta4.json'), 'w'))

    ss = pd.read_csv(os.path.join(root, 'data/sample_submission.csv'))
    ds = Cassava(ss[['image_id']], os.path.join(root, 'data/test_images'), args.size, False)
    ld = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    p, _, ids = infer(model, ld)
    np.save(os.path.join(args.out, 'test_probs_tta4.npy'), p)
    pd.DataFrame({'image_id': ids}).to_csv(os.path.join(args.out, 'test_ids.csv'), index=False)
    print(f'TEST tta4 done -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
