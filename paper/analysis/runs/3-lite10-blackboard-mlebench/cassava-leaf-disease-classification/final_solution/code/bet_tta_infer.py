import argparse, os
import numpy as np, pandas as pd
from torch.utils.data import DataLoader
import torch, cv2, timm
from train import CassavaDS, DATA

cv2.setNumThreads(2)


def infer(model, df, img_dir, scales, batch, workers, device, four_flip):
    flips = [(False, False), (True, False), (False, True), (True, True)] if four_flip else [(False, False), (True, False)]
    pass_preds = []
    for sc in scales:
        ds = CassavaDS(df.assign(label=0) if 'label' not in df else df, img_dir, sc, False)
        ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
        for hf, vf in flips:
            preds = []
            with torch.no_grad():
                for x, _ in ld:
                    x = x.to(device, non_blocking=True)
                    dims = []
                    if hf: dims.append(3)
                    if vf: dims.append(2)
                    xx = torch.flip(x, dims) if dims else x
                    with torch.cuda.amp.autocast():
                        p = torch.softmax(model(xx).float(), 1)
                    preds.append(p.cpu().numpy())
            pass_preds.append(np.concatenate(preds, 0))
    return np.mean(np.stack(pass_preds, 0), 0)


def load(model_name, ckpt, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    ck = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--honest-ckpt', required=True)
    ap.add_argument('--ft-ckpt', default='')
    ap.add_argument('--scales', required=True)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--four-flip', action='store_true')
    args = ap.parse_args()
    device = 'cuda'
    scales = [int(s) for s in args.scales.split(',')]
    os.makedirs(args.outdir, exist_ok=True)

    cv = pd.read_csv('/download_job_code/payload/sandbox/workspaces/aldebaran_epsilon/cv_split.csv')
    va = cv[cv.fold == 0].reset_index(drop=True); y = va.label.values
    sample = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))[['image_id']]

    m = load(args.model, args.honest_ckpt, device)
    vp = infer(m, va, os.path.join(DATA, 'train_images'), scales, args.batch, args.workers, device, args.four_flip)
    print(f'honest val acc={ (vp.argmax(1)==y).mean():.5f} scales={scales} fourflip={args.four_flip}', flush=True)
    np.save(os.path.join(args.outdir, 'val_probs_tta4.npy'), vp)
    tp = infer(m, sample, os.path.join(DATA, 'test_images'), scales, args.batch, args.workers, device, args.four_flip)
    np.save(os.path.join(args.outdir, 'test_probs_tta4.npy'), tp)
    print('honest test done', flush=True)
    if args.ft_ckpt and os.path.exists(args.ft_ckpt):
        mf = load(args.model, args.ft_ckpt, device)
        tf = infer(mf, sample, os.path.join(DATA, 'test_images'), scales, args.batch, args.workers, device, args.flip_ft if hasattr(args, 'flip_ft') else args.four_flip)
        np.save(os.path.join(args.outdir, 'test_probs_ft_tta4.npy'), tf)
        print('FT test done', flush=True)


if __name__ == '__main__':
    main()
