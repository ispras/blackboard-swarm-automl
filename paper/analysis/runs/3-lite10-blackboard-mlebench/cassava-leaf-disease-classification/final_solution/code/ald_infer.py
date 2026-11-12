import argparse, os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Cassava, seed_all, run_eval
import timm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_tta', action='store_true')
    args = ap.parse_args()
    seed_all(42)
    root = '/download_job_code/payload/sandbox'
    os.makedirs(args.out, exist_ok=True)
    model = timm.create_model(args.model, pretrained=False, num_classes=5).to('cuda')
    model.load_state_dict(torch.load(os.path.join(args.src, 'best.pt')))
    model.eval()
    tta = not args.no_tta

    spl = pd.read_csv(os.path.join(root, 'workspaces/aldebaran_epsilon/cv_split.csv'))
    va = spl[spl.fold == 0].reset_index(drop=True)
    va_ds = Cassava(va, os.path.join(root, 'data/train_images'), args.size, False)
    va_ld = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    p, yy, _ = run_eval(model, va_ld, 'cuda', tta=tta)
    acc = (p.argmax(1) == yy).mean()
    np.save(os.path.join(args.out, 'val_probs.npy'), p)
    va[['image_id']].to_csv(os.path.join(args.out, 'val_ids.csv'), index=False)

    ss = pd.read_csv(os.path.join(root, 'data/sample_submission.csv'))
    te_ds = Cassava(ss[['image_id']], os.path.join(root, 'data/test_images'), args.size, False)
    te_ld = DataLoader(te_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    tp, _, ids = run_eval(model, te_ld, 'cuda', tta=tta)
    np.save(os.path.join(args.out, 'test_probs.npy'), tp)
    pd.DataFrame({'image_id': ids}).to_csv(os.path.join(args.out, 'test_ids.csv'), index=False)
    json.dump({'fold0_acc': float(acc), 'tta': tta, 'model': args.model, 'size': args.size},
              open(os.path.join(args.out, 'meta.json'), 'w'), indent=2)
    print('INFER DONE', args.out, 'fold0_acc', acc, flush=True)


if __name__ == '__main__':
    main()
