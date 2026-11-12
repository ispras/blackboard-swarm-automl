import os, sys, json, argparse, time
import numpy as np, pandas as pd, torch, cv2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import GISegDataset, get_mask_map, build_hw_dict

ROOT = "/download_job_code/payload/sandbox"
WS = f"{ROOT}/workspaces/aldebaran_omicron"
CACHE = f"{WS}/cache"


def build_model(ck):
    a = ck["args"]
    model = getattr(smp, a.get("arch", "Unet"))(
        encoder_name=a["encoder"], encoder_weights=None,
        in_channels=a["stack"], classes=3)
    model.load_state_dict(ck["state"])
    return model.cuda().eval()


@torch.no_grad()
def predict(model, dl, stack, tta=True):
    outs = []
    for x, _ in dl:
        x = x.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            p = torch.sigmoid(model(x).float())
            if tta:
                for dims in [(3,), (2,), (2, 3)]:
                    p = p + torch.sigmoid(model(torch.flip(x, dims=dims)).float()).flip(dims=dims)
                p = p / 4.0
        outs.append(p.cpu())
    return torch.cat(outs, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--only_val_cases", default="")
    ap.add_argument("--max_n", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    model = build_model(ck)
    meta = pd.read_csv(f"{CACHE}/{args.split}_index.csv")
    img = np.load(f"{CACHE}/{args.split}_img320.npy", mmap_mode="r")
    hw = build_hw_dict(args.split)
    mask_map = get_mask_map() if args.split == "train" else pd.DataFrame(columns=["large_bowel", "small_bowel", "stomach"])
    idx = np.arange(len(meta))
    if args.only_val_cases:
        vc = set(json.load(open(args.only_val_cases))["val_cases"])
        idx = np.where(meta["case"].isin(vc).values)[0]
    if args.max_n > 0:
        idx = idx[:args.max_n]
    ds = GISegDataset(meta, img, mask_map, hw, a["size"], a["stack"], False, idx)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0)
    t0 = time.time()
    probs = predict(model, dl, a["stack"], tta=bool(args.tta))
    print("infer", probs.shape, time.time() - t0, flush=True)
    mm = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.uint8,
                                   shape=probs.shape)
    mm[:] = (probs * 255.0).round().clip(0, 255).astype(np.uint8)
    mm.flush()
    meta.iloc[idx].to_csv(args.out.replace(".npy", "_index.csv"), index=False)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
