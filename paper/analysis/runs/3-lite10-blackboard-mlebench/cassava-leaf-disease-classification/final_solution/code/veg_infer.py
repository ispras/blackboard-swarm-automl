import argparse, os, json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import timm
import train as T

ROOT = T.ROOT; DATA = T.DATA

def build(model_name, drop_path=0.0):
    m = timm.create_model(model_name, pretrained=False, num_classes=5, drop_path_rate=drop_path)
    return m

@torch.no_grad()
def predict_multiscale(model, df, img_dir, size, device, scales=(1.0,), bs=32, workers=8):
    acc = None
    for sc in scales:
        s = int(round(size*sc))
        ds = T.CassavaDS(df, img_dir, s, train=False)
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
        model.eval(); outs = []
        for x, _ in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                lo = model(x)
                lo = (lo + model(torch.flip(x, dims=[3]))) / 2
            outs.append(torch.softmax(lo.float(), 1).cpu().numpy())
        p = np.concatenate(outs)
        acc = p if acc is None else acc + p
    return acc/len(scales)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--ckpt', default='best.pth')
    ap.add_argument('--scales', default='1.0')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tta-val', action='store_true')
    args = ap.parse_args()
    a = json.load(open(os.path.join(args.run, 'args.json')))
    device = 'cuda'
    model = build(a['model'], a.get('drop_path', 0.0)).to(device)
    ck = torch.load(os.path.join(args.run, args.ckpt), map_location='cpu')
    model.load_state_dict(ck['model'])
    scales = tuple(float(x) for x in args.scales.split(','))
    d = pd.read_csv(a['split'])
    va = d[d.fold == a['fold']].reset_index(drop=True)
    va.to_csv(os.path.join(args.run, 'val_ids.csv'), index=False)
    vp = predict_multiscale(model, va, os.path.join(DATA, 'train_images'), a['size'], device, scales, args.bs, args.workers)
    np.save(os.path.join(args.run, 'val_probs_tta.npy'), vp)
    vacc = (vp.argmax(1) == va.label.values).mean()
    open(os.path.join(args.run, 'val_acc_tta.txt'), 'w').write(str(vacc))
    print('val_acc_tta', vacc, flush=True)
    te = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))
    tp = predict_multiscale(model, te, os.path.join(DATA, 'test_images'), a['size'], device, scales, args.bs, args.workers)
    np.save(os.path.join(args.run, 'test_probs_tta.npy'), tp)
    te.assign(label=tp.argmax(1)).to_csv(os.path.join(args.run, 'submission_tta.csv'), index=False)
    print('done', args.run, flush=True)

if __name__ == '__main__':
    main()
