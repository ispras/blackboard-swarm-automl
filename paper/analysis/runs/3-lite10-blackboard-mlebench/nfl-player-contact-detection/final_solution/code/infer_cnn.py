import os, sys, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cnn import Net, normalize, OUT, S


class TestDS(Dataset):
    def __init__(self, split):
        self.arr = np.load(f"{OUT}/{split}_crops.npy", mmap_mode="r")
        self.geom = np.load(f"{OUT}/geom_{split}.npy", mmap_mode="r")
        self.n = len(self.geom)

    def __len__(self):
        return self.n

    def __getitem__(self, r):
        x = np.ascontiguousarray(np.asarray(self.arr[r]).transpose(0, 3, 1, 2))
        return torch.from_numpy(x), torch.from_numpy(np.asarray(self.geom[r]).copy())


@torch.no_grad()
def pred_model(model, dl, device, flip=False):
    model.eval()
    out = []
    for x, g in dl:
        x = x.to(device, non_blocking=True)
        if flip:
            x = x.flip(-1)
        x = normalize(x)
        g = g.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            z = model(x, g)
        out.append(torch.sigmoid(z.float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    folds = [int(f) for f in args.folds.split(",")]
    device = "cuda"
    plan = pd.read_parquet(f"{OUT}/{args.split}_cand.parquet")
    ds = TestDS(args.split)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    preds = []
    for f in folds:
        model = Net("resnet18", ds.geom.shape[1]).to(device)
        import os
        wpath = f"{OUT}/model_f{f}_{args.tag}_best.pt"
        if not os.path.exists(wpath):
            wpath = f"{OUT}/model_f{f}_{args.tag}.pt"
        model.load_state_dict(torch.load(wpath, map_location=device))
        p = 0.5 * (pred_model(model, dl, device, False) + pred_model(model, dl, device, True))
        preds.append(p)
        print(f"fold {f} done", flush=True)
    p = np.mean(preds, 0)
    out = args.out or f"{OUT}/{args.split}_pred_{args.tag}.parquet"
    pd.DataFrame({"contact_id": plan.contact_id.values, "pred": p}).to_parquet(out, index=False)
    print("saved", out, len(p), p.mean())


if __name__ == "__main__":
    main()
