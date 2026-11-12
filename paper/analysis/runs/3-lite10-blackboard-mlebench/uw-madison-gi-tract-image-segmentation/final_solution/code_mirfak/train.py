import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import metric as MT

CACHE = D.CACHE
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="UnetPlusPlus")
    p.add_argument("--encoder", default="efficientnet-b3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--fold", type=int, default=0, help="-1 = train on all cases")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="exp")
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--init_ckpt", default="", help="load weights from this checkpoint before training")
    return p.parse_args()


def build_model(arch, encoder):
    return smp.create_model(arch, encoder_name=encoder, encoder_weights="imagenet",
                            in_channels=3, classes=3)


def validate(model, val_samples, meta, val_cases, device, bs=64, compute_combo=True):
    model.eval()
    # group samples by scan
    by_scan = {}
    for mi, k in val_samples:
        by_scan.setdefault(mi, []).append(k)
    pred_vol = {}
    for mi, ks in by_scan.items():
        m = meta[mi]
        pred_vol[mi] = np.zeros((m["n"], 3, D.IMG_SIZE, D.IMG_SIZE), np.uint8)
    # dataloader without aug
    ds = D.GISDataset("train", val_samples, meta, train=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    ptr = 0
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
            prob = torch.sigmoid(logits).float().cpu().numpy()
            for j in range(prob.shape[0]):
                mi, k = val_samples[ptr + j]
                pred_vol[mi][k] = (prob[j] > 0.5).astype(np.uint8)
            ptr += prob.shape[0]
    dices = []; hds = []
    for mi, ks in by_scan.items():
        m = meta[mi]
        gt = np.load(os.path.join(CACHE, m["msk"]), mmap_mode="r")
        gt_vol = (np.asarray(gt) > 0).astype(np.uint8).transpose(3, 0, 1, 2)  # 3,N,H,W
        pv = pred_vol[mi].transpose(1, 0, 2, 3)  # (N,3,H,W)->(3,N,H,W)
        for c in range(3):
            # dice per-slice (grader averages over (id,class) rows)
            for z in range(pv.shape[1]):
                dices.append(MT.dice_bin(pv[c, z], gt_vol[c, z]))
            if compute_combo:
                hds.append(MT.hausdorff_norm(pv[c], gt_vol[c]))
    dice = float(np.mean(dices))
    if compute_combo:
        hd = float(np.mean(hds)); combo = 0.4 * dice + 0.6 * (1 - hd)
    else:
        hd = float("nan"); combo = float("nan")
    return dice, hd, combo


def main():
    a = get_args()
    os.makedirs(OUT, exist_ok=True)
    run_dir = os.path.join(OUT, a.name)
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    folds = D.get_folds()
    meta = D.load_meta("train")
    all_cases = sorted({m["case"] for m in meta})
    if a.fold >= 0:
        val_cases = {c for c in all_cases if folds[c] == a.fold}
    else:
        val_cases = set()
    tr_cases = set(all_cases) - val_cases
    _, tr_samples = D.build_samples("train", cases=tr_cases)
    _, val_samples = D.build_samples("train", cases=val_cases) if val_cases else (None, [])
    print(f"fold {a.fold}: train cases {len(tr_cases)}, val cases {len(val_cases)}, train slices {len(tr_samples)}, val slices {len(val_samples)}", flush=True)

    tr_ds = D.GISDataset("train", tr_samples, meta, train=True, seed=a.seed)
    tr_dl = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=(a.workers > 0))

    model = build_model(a.arch, a.encoder).to(device)
    if a.init_ckpt:
        ck = torch.load(a.init_ckpt, map_location="cpu")
        sd = ck["model"] if "model" in ck else ck
        model.load_state_dict(sd)
        print(f"initialized from {a.init_ckpt}", flush=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    raw = model.module if hasattr(model, "module") else model

    bce = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="multilabel", from_logits=True)

    def criterion(logits, target):
        return 0.5 * bce(logits, target) + 0.5 * dice_loss(logits, target)

    decay, no_decay = [], []
    for n, p in raw.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    steps = len(tr_dl) * a.epochs
    warm = len(tr_dl) * a.warmup

    def lr_lambda(step):
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, steps - warm)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    best = -1
    hist = []
    for ep in range(a.epochs):
        model.train()
        t0 = time.time(); tot = 0; nb = 0
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1
        tr_loss = tot / max(1, nb)
        msg = f"ep {ep+1}/{a.epochs} loss {tr_loss:.4f} lr {opt.param_groups[0]['lr']:.2e} time {time.time()-t0:.0f}s"
        rec = {"epoch": ep + 1, "train_loss": tr_loss, "time": time.time() - t0}
        if val_cases and ((ep + 1) % a.val_every == 0 or ep == a.epochs - 1):
            dice, hd, combo = validate(model, val_samples, meta, val_cases, device)
            rec.update({"val_dice": dice, "val_hd": hd, "val_combo": combo})
            msg += f" | val_dice {dice:.4f} hd {hd:.4f} combo {combo:.4f}"
            score = combo if not np.isnan(combo) else dice
            if score > best:
                best = score
                torch.save({"model": raw.state_dict(), "args": vars(a), "epoch": ep + 1,
                            "val_dice": dice, "val_combo": combo}, os.path.join(run_dir, "best.pt"))
                msg += " *saved*"
        hist.append(rec)
        print(msg, flush=True)
        with open(os.path.join(run_dir, "history.json"), "w") as fh:
            json.dump(hist, fh, indent=1)
    torch.save({"model": raw.state_dict(), "args": vars(a), "epoch": a.epochs},
               os.path.join(run_dir, "last.pt"))
    print("DONE", run_dir, "best", best, flush=True)


if __name__ == "__main__":
    main()
