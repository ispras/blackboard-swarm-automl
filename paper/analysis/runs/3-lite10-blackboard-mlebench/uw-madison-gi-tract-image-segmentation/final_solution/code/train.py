import os, sys, time, argparse, json
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp

ROOT = "/download_job_code/payload/sandbox"
WS = f"{ROOT}/workspaces/capella_zeta"
CACHE = f"{WS}/cache"
CLASSES = ["large_bowel", "small_bowel", "stomach"]
S = 320


def make_index():
    idx = pd.read_csv(f"{CACHE}/train_index.csv")
    idx = idx.sort_values(["case_day", "slice"]).reset_index(drop=True)
    idx["row"] = np.arange(len(idx))
    prev = idx["row"].values.copy(); nxt = idx["row"].values.copy()
    prev2 = idx["row"].values.copy(); nxt2 = idx["row"].values.copy()
    for _, g in idx.groupby("case_day"):
        pos = g.index.values
        rows = idx.loc[pos, "row"].values
        prev[pos[0]] = rows[0]; nxt[pos[-1]] = rows[-1]
        prev[pos[1:]] = rows[:-1]; nxt[pos[:-1]] = rows[1:]
        prev2[pos[:2]] = rows[0]; prev2[pos[2:]] = rows[:-2]
        nxt2[pos[-2:]] = rows[-1]; nxt2[pos[:-2]] = rows[2:]
    idx["prev_row"] = prev; idx["next_row"] = nxt
    idx["prev2_row"] = prev2; idx["next2_row"] = nxt2
    cases = sorted(idx["case"].unique())
    rng = np.random.RandomState(42); perm = rng.permutation(len(cases))
    fold = {c: i % 5 for i, c in enumerate(np.asarray(cases)[perm])}
    idx["fold"] = idx["case"].map(fold)
    return idx


def tf_train():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=15,
                           border_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([A.GaussNoise(var_limit=(5, 30)),
                 A.GaussianBlur(blur_limit=(3, 5)),
                 A.MotionBlur(blur_limit=5)], p=0.2),
    ])


class GIDataset(Dataset):
    def __init__(self, df, train=True, n_slices=3):
        self.df = df.reset_index(drop=True)
        self.img = np.load(f"{CACHE}/train_img{S}.npy", mmap_mode="r")
        self.msk = np.load(f"{CACHE}/train_mask{S}.npy", mmap_mode="r")
        self.train = train
        self.n_slices = n_slices
        self.tf = tf_train() if train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, p):
        r = self.df.iloc[p]
        c = int(r["row"])
        if self.n_slices == 5:
            rows = [int(r["prev2_row"]), int(r["prev_row"]), c, int(r["next_row"]), int(r["next2_row"])]
        else:
            rows = [int(r["prev_row"]), c, int(r["next_row"])]
        img = np.stack([self.img[i] for i in rows], axis=-1)
        msk = self.msk[c].transpose(1, 2, 0).astype(np.float32)
        if self.tf is not None:
            o = self.tf(image=img, mask=msk); img = o["image"]; msk = o["mask"]
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        msk = msk.transpose(2, 0, 1)
        return torch.from_numpy(img), torch.from_numpy(msk)


def per_sample_dice(pr, y, eps=1e-6):
    # pr,y: [B,C,H,W] binary float; dice with both-empty -> 1
    inter = (pr * y).sum((2, 3))
    den = pr.sum((2, 3)) + y.sum((2, 3))
    d = torch.where(den > 0, 2 * inter / (den + eps), torch.ones_like(den))
    return d  # [B,C]


def build_model(arch, encoder, in_ch=3):
    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=in_ch, classes=3, activation=None)
    if arch == "unetpp":
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights="imagenet", in_channels=in_ch, classes=3, activation=None)
    if arch == "deeplab":
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet", in_channels=in_ch, classes=3, activation=None)
    raise ValueError(arch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--arch", type=str, default="unet")
    p.add_argument("--encoder", type=str, default="mit_b3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tag", type=str, default="mitb3_f0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_slices", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--val_limit", type=int, default=0)
    p.add_argument("--save_oof", action="store_true")
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--resume", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    OUTD = f"{WS}/ckpt"; os.makedirs(OUTD, exist_ok=True)
    OOFD = f"{WS}/oof"; os.makedirs(OOFD, exist_ok=True)

    idx = make_index()
    if args.fold < 0:
        tr = idx.reset_index(drop=True); va = None
    else:
        tr = idx[idx.fold != args.fold].reset_index(drop=True)
        va = idx[idx.fold == args.fold].reset_index(drop=True)
        if args.val_limit:
            va = va.iloc[:args.val_limit].reset_index(drop=True)
    trds = GIDataset(tr, train=True, n_slices=args.n_slices)
    trl = DataLoader(trds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                     drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val = None
    if va is not None:
        vads = GIDataset(va, train=False, n_slices=args.n_slices)
        val = DataLoader(vads, batch_size=args.bs * 2, shuffle=False, num_workers=args.workers,
                         pin_memory=True, persistent_workers=True)
    print(f"train {len(trds)} val {0 if va is None else len(vads)} steps/epoch {len(trl)}", flush=True)

    model = build_model(args.arch, args.encoder, args.n_slices).to(dev)
    if args.resume:
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        print("resumed", args.resume, flush=True)
    dice_loss = smp.losses.DiceLoss(mode="multilabel", smooth=1.0)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(trl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    scaler = GradScaler()

    @torch.no_grad()
    def evaluate(save_probs=False):
        model.eval()
        dsum = np.zeros(3); dn = 0; pos_sum = np.zeros(3); pos_n = np.zeros(3)
        probs_all = []
        for x, y in val:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            with autocast():
                lo = model(x)
            prob = torch.sigmoid(lo)
            pr = (prob > 0.5).float()
            d = per_sample_dice(pr, y)  # [B,C]
            dsum += d.sum(0).cpu().numpy(); dn += x.shape[0]
            pos = (y.sum((2, 3)) > 0).float()
            pos_sum += (d * pos).sum(0).cpu().numpy(); pos_n += pos.sum(0).cpu().numpy()
            if save_probs:
                probs_all.append(prob.half().cpu().numpy())
        model.train()
        macro = dsum / max(dn, 1)
        posdice = pos_sum / np.maximum(pos_n, 1)
        return macro, posdice, dn, pos_n, (np.concatenate(probs_all, 0) if save_probs else None)

    best = -1; step = 0; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); run = 0.0; nb = 0; t1 = time.time()
        for x, y in trl:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                lo = model(x)
                loss = 0.5 * bce(lo, y) + 0.5 * dice_loss(lo, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); nb += 1; step += 1
            if args.max_steps and step >= args.max_steps:
                break
            if nb % 100 == 0:
                print(f"ep{ep} {nb}/{len(trl)} loss {run/nb:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time()-t1)/nb:.2f}s/it", flush=True)
        if val is None:
            print(f"EPOCH {ep} loss {run/max(nb,1):.4f} {time.time()-t1:.0f}s (all-data)", flush=True)
            if not args.no_save:
                torch.save({"model": model.state_dict(), "arch": args.arch, "encoder": args.encoder, "n_slices": args.n_slices,
                            "fold": -1, "epoch": ep}, f"{OUTD}/{args.tag}_last.pt")
            if args.max_steps and step >= args.max_steps:
                break
            continue
        macro, posdice, n, npos, _ = evaluate()
        print(f"EPOCH {ep} loss {run/max(nb,1):.4f} {time.time()-t1:.0f}s | macro {macro.round(4)} "
              f"mean {macro.mean():.4f} | pos {posdice.round(4)} mean {posdice.mean():.4f} n {n}", flush=True)
        score = macro.mean()
        if score > best and not args.no_save:
            best = score
            torch.save({"model": model.state_dict(), "arch": args.arch, "encoder": args.encoder, "n_slices": args.n_slices,
                        "fold": args.fold, "macro": macro.tolist(), "pos": posdice.tolist(),
                        "score": score, "epoch": ep}, f"{OUTD}/{args.tag}_best.pt")
            print(f"  saved best {best:.4f}", flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    if not args.no_save:
        torch.save({"model": model.state_dict(), "arch": args.arch, "encoder": args.encoder, "n_slices": args.n_slices,
                    "fold": args.fold, "score": best}, f"{OUTD}/{args.tag}_last.pt")
    if args.save_oof and va is not None:
        path = args.resume or f"{OUTD}/{args.tag}_best.pt"
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        macro, posdice, n, npos, probs = evaluate(save_probs=True)
        np.save(f"{OOFD}/{args.tag}_probs.npy", probs)
        va[["id", "case", "case_day", "slice", "H", "W", "fold"]].to_csv(f"{OOFD}/{args.tag}_ids.csv", index=False)
        print("OOF saved", args.tag, macro, flush=True)
    print("DONE", args.tag, "best", best, "elapsed", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
