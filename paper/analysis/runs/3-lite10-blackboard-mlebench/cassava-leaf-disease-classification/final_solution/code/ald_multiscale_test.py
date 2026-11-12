import argparse, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Cassava, seed_all
import timm


def infer(model, ld):
    model.eval(); out = []
    with torch.no_grad():
        for x, _ in ld:
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
                p = (acc / 4.0).float()
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--sizes', default='416,448,480,512')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    seed_all(42)
    root = '/download_job_code/payload/sandbox'
    ss = pd.read_csv(os.path.join(root, 'data/sample_submission.csv'))
    state = torch.load(args.ckpt)
    acc = None
    for s in [int(x) for x in args.sizes.split(',')]:
        model = timm.create_model(args.model, pretrained=False, num_classes=5, img_size=s).to('cuda')
        model.load_state_dict(state, strict=False); model.eval()
        ds = Cassava(ss[['image_id']], os.path.join(root, 'data/test_images'), s, False)
        ld = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
        p = infer(model, ld); acc = p if acc is None else acc + p
        print(f'size {s} done', flush=True)
        del model; torch.cuda.empty_cache()
    p = acc / len([int(x) for x in args.sizes.split(',')])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, p)
    pd.DataFrame({'image_id': ss.image_id}).to_csv(os.path.join(os.path.dirname(args.out), 'test_ids.csv'), index=False)
    print('saved', args.out, p.shape, flush=True)


if __name__ == '__main__':
    main()
