import argparse, os, time, json
import numpy as np, pandas as pd
from torch.utils.data import DataLoader
import torch, torch.nn as nn
import timm

from train import CassavaDS, Mixup, set_seed, predict_tta, DATA, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--img-size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=4e-5)
    ap.add_argument('--min-lr', type=float, default=1e-6)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--out', required=True)
    ap.add_argument('--drop-path', type=float, default=0.1)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'ft_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    df = pd.read_csv(os.path.join(DATA, 'train.csv'))
    print(f'full train={len(df)}', flush=True)
    tr_ds = CassavaDS(df, os.path.join(DATA, 'train_images'), args.img_size, True)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=2)

    device = 'cuda'
    model = timm.create_model(args.model, pretrained=False, num_classes=5, drop_path_rate=args.drop_path)
    ck = torch.load(args.init, map_location='cpu')
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    model.load_state_dict(sd)
    model = model.to(device)
    print(f'loaded init acc={ck.get("acc") if isinstance(ck, dict) else "?"}', flush=True)

    decay, no_decay = [], []
    for n_, p_ in model.named_parameters():
        (no_decay if (p_.ndim == 1 or n_.endswith('.bias')) else decay).append(p_)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.wd},
                             {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr)
    steps = len(tr_ld) * args.epochs
    warm = len(tr_ld) // 2
    def _lam(s):
        if s < warm:
            return s / max(1, warm)
        prog = (s - warm) / max(1, steps - warm)
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lam)
    scaler = torch.cuda.amp.GradScaler()
    mix = Mixup(prob=0.4, smooth=0.1)

    model.train()
    for ep in range(args.epochs):
        tl = 0.0; t0 = time.time()
        for x, y in tr_ld:
            x = x.to(device, non_blocking=True); y = y.to(device)
            xm, ym = mix(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                out = model(xm)
                loss = torch.sum(-ym * torch.log_softmax(out.float(), 1), 1).mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        print(f'ft epoch {ep+1}/{args.epochs} loss {tl/len(tr_ld):.4f} lr {sched.get_last_lr()[0]:.2e} time {time.time()-t0:.0f}s', flush=True)
    torch.save({'model': model.state_dict()}, os.path.join(args.out, 'ft_last.pt'))

    te = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))[['image_id']]
    te_ds = CassavaDS(te.assign(label=0), os.path.join(DATA, 'test_images'), args.img_size, False)
    te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    tp = predict_tta(model, te_ld, device, args.img_size)
    np.save(os.path.join(args.out, 'test_probs_ft.npy'), tp)
    print('saved', os.path.join(args.out, 'test_probs_ft.npy'), flush=True)


if __name__ == '__main__':
    main()
