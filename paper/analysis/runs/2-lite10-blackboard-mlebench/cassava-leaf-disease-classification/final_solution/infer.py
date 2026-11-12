import os, sys, argparse, glob, time
import numpy as np, pandas as pd
from PIL import Image
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths = paths; self.labels = labels; self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]).convert('RGB'))
        return self.tf(image=img)['image'], self.labels[i]


def tfm(size):
    return A.Compose([A.Resize(size, size), A.Normalize(MEAN, STD), ToTensorV2()])


@torch.no_grad()
def run(model, paths, labels, size, bs, workers, scales, flips, rots=(0,)):
    ds = DS(paths, labels, tfm(size))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    ps = []
    for x, y in dl:
        x = x.cuda(non_blocking=True)
        o = 0
        n = 0
        for s in scales:
            if s != 1.0:
                xs = torch.nn.functional.interpolate(x, scale_factor=s, mode='bilinear',
                                                    align_corners=False)
            else:
                xs = x
            for r in rots:
                xr = torch.rot90(xs, k=r // 90, dims=[2, 3]) if r else xs
                for f in flips:
                    z = xr
                    if 'h' in f:
                        z = torch.flip(z, dims=[3])
                    if 'v' in f:
                        z = torch.flip(z, dims=[2])
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        o = o + model(z).float()
                    n += 1
        ps.append(torch.softmax(o / n, 1).cpu().numpy())
    return np.concatenate(ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--out', default='')
    p.add_argument('--model', default='convnext_tiny')
    p.add_argument('--img-size', type=int, default=384)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--scales', default='1.0')
    p.add_argument('--flips', default='h')
    p.add_argument('--rots', default='0')
    a = p.parse_args()
    size = a.img_size
    ck = torch.load(a.ckpt, map_location='cpu')
    model = timm.create_model(a.model, pretrained=False, num_classes=5).cuda()
    model.load_state_dict(ck['model']); model.eval()
    df = pd.read_csv(os.path.join(DATA, 'train.csv'))
    df['fold'] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for f, (_, vi) in enumerate(skf.split(df, df.label)):
        df.loc[vi, 'fold'] = f
    val = df[df.fold == a.fold].reset_index(drop=True)
    test = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))
    scales = [float(x) for x in a.scales.split(',')]
    flips = a.flips.split(',')
    rots = [int(x) for x in a.rots.split(',')]
    t0 = time.time()
    vp = run(model, [os.path.join(DATA, 'train_images', x) for x in val.image_id],
             val.label.values, size, a.batch, a.workers, scales, flips, rots)
    acc = (vp.argmax(1) == val.label.values).mean()
    tp = run(model, [os.path.join(DATA, 'test_images', x) for x in test.image_id],
             np.zeros(len(test), dtype=int), size, a.batch, a.workers, scales, flips, rots)
    print(f'fold {a.fold} tta acc {acc:.4f} time {time.time()-t0:.1f}s', flush=True)
    out = a.out or os.path.dirname(a.ckpt)
    tag = a.ckpt.split('/')[-1].replace('.pth', '')
    pd.DataFrame({'image_id': val.image_id, **{f'p{i}': vp[:, i] for i in range(5)},
                  'label': val.label.values}).to_csv(
        os.path.join(out, f'val_probs_{tag}_tta.csv'), index=False)
    pd.DataFrame({'image_id': test.image_id, **{f'p{i}': tp[:, i] for i in range(5)}}).to_csv(
        os.path.join(out, f'test_probs_{tag}_tta.csv'), index=False)


if __name__ == '__main__':
    main()
