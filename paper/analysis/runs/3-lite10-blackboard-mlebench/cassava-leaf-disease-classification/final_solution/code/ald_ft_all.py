import argparse, os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Cassava, seed_all, run_eval, mixup_cutmix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)      # dir with best.pt
    ap.add_argument('--model', required=True)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    seed_all(42)
    root = '/download_job_code/payload/sandbox'
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, 'args.json'), 'w'))

    tr = pd.read_csv(os.path.join(root, 'data/train.csv'))
    if args.smoke:
        tr = tr.sample(200, random_state=0).reset_index(drop=True)
    print('all-data train', len(tr), flush=True)
    ds = Cassava(tr, os.path.join(root, 'data/train_images'), args.size, True)
    ld = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    model = timm_create(args.model).to('cuda')
    model.load_state_dict(torch.load(os.path.join(args.src, 'best.pt')))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = len(ld); total = args.epochs * steps
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * s / max(1, total))))
    scaler = torch.cuda.amp.GradScaler()
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); losses = []
        for i, (x, y) in enumerate(ld):
            x = x.to('cuda', non_blocking=True); y = y.to('cuda', non_blocking=True)
            if np.random.rand() < 0.5:
                x, ya, yb, lam = mixup_cutmix(x, y)
                logits = model(x)
                loss = crit(logits, ya) * lam + crit(logits, yb) * (1 - lam)
            else:
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sch.step()
            losses.append(loss.item())
            if (i + 1) % 100 == 0:
                print(f'  ep{ep} step {i+1}/{steps} loss {np.mean(losses[-100:]):.4f} t {time.time()-t0:.0f}s', flush=True)
        print(f'FT ep {ep+1}/{args.epochs} loss {np.mean(losses):.4f} t {time.time()-t0:.0f}s', flush=True)
    torch.save(model.state_dict(), os.path.join(args.out, 'ft.pt'))

    ss = pd.read_csv(os.path.join(root, 'data/sample_submission.csv'))
    te_ds = Cassava(ss[['image_id']], os.path.join(root, 'data/test_images'), args.size, False)
    te_ld = DataLoader(te_ds, batch_size=args.bs * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    p, _, ids = run_eval(model, te_ld, 'cuda', tta=True)
    np.save(os.path.join(args.out, 'test_probs_ft.npy'), p)
    pd.DataFrame({'image_id': ids}).to_csv(os.path.join(args.out, 'test_ids.csv'), index=False)
    print('DONE FT', args.out, flush=True)


def timm_create(name):
    import timm
    return timm.create_model(name, pretrained=False, num_classes=5)


if __name__ == '__main__':
    main()
