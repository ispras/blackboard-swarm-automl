import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import GISegDataset, get_mask_map, build_hw_dict, case_split, CLASSES

ROOT = "/download_job_code/payload/sandbox"
WS = f"{ROOT}/workspaces/aldebaran_omicron"
CACHE = f"{WS}/cache"


def dice_metrics(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    dims = (2, 3)
    inter = (pred * target).sum(dims)
    s = pred.sum(dims) + target.sum(dims)
    dice = torch.where(s > 0, 2 * inter / (s + eps), torch.ones_like(s))
    pos = s > 0
    pos_dice = torch.where(pos, dice, torch.zeros_like(dice))
    return dice, pos_dice, pos


class DiceBCELoss(torch.nn.Module):
    def __init__(self, bce_w=0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.bce_w = bce_w

    def forward(self, logits, target):
        b = self.bce(logits, target)
        p = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (p * target).sum(dims)
        s = p.sum(dims) + target.sum(dims)
        d = (2 * inter + 1.0) / (s + 1.0)
        return self.bce_w * b + (1 - self.bce_w) * (1 - d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="efficientnet-b4")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--stack", type=int, default=3)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="modelA")
    ap.add_argument("--train_all", action="store_true")
    ap.add_argument("--val_cases_file", default=f"{ROOT}/workspaces/regulus_alpha/val_cases.json")
    ap.add_argument("--init_ckpt", default="")
    ap.add_argument("--mask_cache_dir", default="")
    args = ap.parse_args()

    torch.manual_seed(42 + args.fold)
    np.random.seed(42 + args.fold)
    outdir = f"{WS}/runs/{args.tag}_f{args.fold}"
    os.makedirs(outdir, exist_ok=True)
    print("args", vars(args), flush=True)

    meta = pd.read_csv(f"{CACHE}/train_index.csv")
    img = np.load(f"{CACHE}/train_img320.npy", mmap_mode="r")
    mask_map = get_mask_map()
    hw = build_hw_dict("train")

    if args.train_all:
        val_cases = set()
    elif os.path.exists(args.val_cases_file):
        val_cases = set(json.load(open(args.val_cases_file))["val_cases"])
    else:
        val_cases = case_split(meta["case"].unique().tolist(), args.val_frac, seed=42, fold=args.fold)
    is_val = meta["case"].isin(val_cases).values
    tr_idx = np.where(~is_val)[0]
    va_idx = np.where(is_val)[0]
    print("train", len(tr_idx), "val", len(va_idx), "val_cases", len(val_cases), flush=True)

    mask_mm = None; id2row = None
    if args.mask_cache_dir:
        import numpy as _np
        cidx = pd.read_csv(f"{args.mask_cache_dir}/index.csv")
        id2row = dict(zip(cidx["id"], range(len(cidx))))
        mask_mm = _np.load(f"{args.mask_cache_dir}/masks_u8.npy", mmap_mode="r")
        print("using mask cache", args.mask_cache_dir, flush=True)
    ds_tr = GISegDataset(meta, img, mask_map, hw, args.size, args.stack, True, tr_idx, mask_mm, id2row)
    ds_va = GISegDataset(meta, img, mask_map, hw, args.size, args.stack, False, va_idx, mask_mm, id2row)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    model = getattr(smp, args.arch)(encoder_name=args.encoder, encoder_weights="imagenet",
                                   in_channels=args.stack, classes=3).cuda()
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(ck["state"])
        print("initialized from", args.init_ckpt, flush=True)
    crit = DiceBCELoss().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(dl_tr) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    best = -1
    log = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); run = 0; nb = 0
        for x, y in dl_tr:
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                o = model(x)
                loss = crit(o.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); nb += 1
        model.eval()
        dsum = torch.zeros(3, device="cuda"); psum = torch.zeros(3, device="cuda")
        pcnt = torch.zeros(3, device="cuda"); n = 0
        with torch.no_grad():
            for x, y in dl_va:
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    o = model(x)
                d, pdice, pos = dice_metrics(o.float(), y)
                dsum += d.sum(0); psum += pdice.sum(0); pcnt += pos.sum(0).float(); n += x.shape[0]
        if n == 0:
            md = np.zeros(3); mp = np.zeros(3)
        else:
            md = (dsum / n).cpu().numpy()
            mp = (psum / pcnt.clamp(min=1)).cpu().numpy()
        score = md.mean()
        rec = dict(epoch=ep, loss=run / max(nb, 1), dice=float(score),
                   dice_per_class=md.tolist(), pos_dice_per_class=mp.tolist(),
                   sec=time.time() - t0)
        print(json.dumps(rec), flush=True)
        log.append(rec)
        if len(va_idx) > 0 and score > best:
            best = score
            torch.save(dict(state=model.state_dict(), args=vars(args), score=score,
                            val_cases=sorted(val_cases)), f"{outdir}/best.pt")
            print("saved best", best, flush=True)
        elif len(va_idx) == 0:
            torch.save(dict(state=model.state_dict(), args=vars(args), score=0,
                            val_cases=[]), f"{outdir}/last.pt")
        with open(f"{outdir}/log.json", "w") as f:
            json.dump(log, f, indent=1)
    if len(va_idx) == 0:
        os.replace(f"{outdir}/last.pt", f"{outdir}/best.pt")
    print("DONE best", best, flush=True)


if __name__ == "__main__":
    main()
