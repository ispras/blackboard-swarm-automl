import os, sys, argparse
import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model, make_index, S, CACHE, WS


class InferDataset(Dataset):
    def __init__(self, df, split, n_slices=3):
        self.df = df.reset_index(drop=True)
        self.img = np.load(f"{CACHE}/{split}_img{S}.npy", mmap_mode="r")
        self.n_slices = n_slices

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
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        return torch.from_numpy(img)


def add_neighbors(df, split):
    df = df.copy()
    if split == "train":
        full = make_index()
        use = full[full["fold"] == df["fold"].iloc[0]] if "fold" in df else full
        return use.reset_index(drop=True)
    df = df.sort_values(["case_day", "slice"]).reset_index(drop=True)
    df["row"] = np.arange(len(df))
    prev = df["row"].values.copy(); nxt = df["row"].values.copy()
    prev2 = df["row"].values.copy(); nxt2 = df["row"].values.copy()
    for _, g in df.groupby("case_day"):
        pos = g.index.values
        rows = df.loc[pos, "row"].values
        prev[pos[0]] = rows[0]; nxt[pos[-1]] = rows[-1]
        prev[pos[1:]] = rows[:-1]; nxt[pos[:-1]] = rows[1:]
        prev2[pos[:2]] = rows[0]; prev2[pos[2:]] = rows[:-2]
        nxt2[pos[-2:]] = rows[-1]; nxt2[pos[:-2]] = rows[2:]
    df["prev_row"] = prev; df["next_row"] = nxt
    df["prev2_row"] = prev2; df["next2_row"] = nxt2
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")  # test | val
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tta", type=str, default="", help="comma list of flips: h,v (e.g. 'h' or 'h,v')")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda"
    sd = torch.load(args.ckpt, map_location="cpu")
    arch = sd.get("arch", "unet"); enc = sd.get("encoder", "efficientnet-b4")
    n_slices = sd.get("n_slices", 3)
    model = build_model(arch, enc, n_slices).to(dev).eval()
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    print("loaded", args.ckpt, arch, enc, flush=True)

    if args.split == "test":
        df = add_neighbors(pd.read_csv(f"{CACHE}/test_index.csv"), "test")
        split = "test"
    else:
        full = make_index()
        df = full[full["fold"] == args.fold].reset_index(drop=True)
        split = "train"
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)

    ds = InferDataset(df, split, n_slices)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                    pin_memory=True, persistent_workers=True)
    out = []
    with torch.no_grad():
        for i, x in enumerate(dl):
            x = x.to(dev, non_blocking=True)
            with autocast():
                p = torch.sigmoid(model(x)).float()
            modes = [m for m in args.tta.split(",") if m]
            n = 1
            for m in modes:
                dims = []
                if "h" in m:
                    dims.append(3)
                if "v" in m:
                    dims.append(2)
                xx = torch.flip(x, dims=dims)
                with autocast():
                    pp = torch.sigmoid(model(xx)).float()
                p = p + torch.flip(pp, dims=dims)
                n += 1
            p = p / n
            out.append(p.half().cpu().numpy())
            if i % 20 == 0:
                print(i, len(dl), flush=True)
    probs = np.concatenate(out, 0)
    od = f"{WS}/preds"; os.makedirs(od, exist_ok=True)
    np.save(f"{od}/{args.tag}_probs.npy", probs)
    cols = ["id", "case", "case_day", "slice"]
    keep = [c for c in cols + ["H", "W"] if c in df.columns]
    df[keep].to_csv(f"{od}/{args.tag}_ids.csv", index=False)
    print("saved", probs.shape, f"{od}/{args.tag}_probs.npy", flush=True)


if __name__ == "__main__":
    main()
