import json, os, argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 107
SCORED = 68
TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_IDX = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C

SEQ_VOCAB = {c: i for i, c in enumerate("ACGU")}
STRUCT_VOCAB = {c: i for i, c in enumerate(".()")}
LOOP_VOCAB = {c: i for i, c in enumerate("SMIBHEX")}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def encode(seq, struct, loop):
    n = len(seq)
    x = np.zeros((n, 14), dtype=np.float32)
    for i, c in enumerate(seq):
        x[i, SEQ_VOCAB.get(c, 0)] = 1.0
    for i, c in enumerate(struct):
        x[i, 4 + STRUCT_VOCAB.get(c, 0)] = 1.0
    for i, c in enumerate(loop):
        x[i, 7 + LOOP_VOCAB.get(c, 0)] = 1.0
    return x


def reverse_comp(x, y):
    # x: (L,14), y: (L,5). reverse seq order; swap structure '(' <-> ')'
    xr = x[::-1].copy()
    xr[:, 4], xr[:, 5] = x[::-1, 5], x[::-1, 4]
    return xr, y[::-1].copy()


def reverse_x(x):
    xr = x[::-1].copy()
    xr[:, 4], xr[:, 5] = x[::-1, 5], x[::-1, 4]
    return xr


def predict_batch(model, xs, dev, tta=False):
    import torch as _t
    model.eval()
    with _t.no_grad():
        p = model(_t.from_numpy(xs).to(dev)).float().cpu().numpy()
        if tta:
            xr = np.stack([reverse_x(x) for x in xs])
            pr = model(_t.from_numpy(xr).to(dev)).float().cpu().numpy()
            pr = pr[:, ::-1, :]
            p = (p + pr) / 2.0
    return p


class RNADataset(Dataset):
    def __init__(self, items, train=False, use_all_targets=True, clip_aux=0.0, aug=True,
                 asinh_s=0.0):
        self.items = items
        self.train = train
        self.use_all_targets = use_all_targets
        self.clip_aux = clip_aux
        self.aug = aug
        self.asinh_s = asinh_s

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        x = encode(it["sequence"], it["structure"], it["predicted_loop_type"])
        y = np.zeros((SEQ_LEN, 5), dtype=np.float32)
        if "reactivity" in it:
            for k, t in enumerate(TARGETS):
                if self.use_all_targets or k in SCORED_IDX:
                    v = np.asarray(it[t], dtype=np.float32)
                    if self.clip_aux > 0 and k not in SCORED_IDX:
                        v = np.clip(v, -self.clip_aux, self.clip_aux)
                    y[:SCORED, k] = v
        if self.asinh_s > 0:
            y[:SCORED] = np.arcsinh(y[:SCORED] / self.asinh_s)
        if self.train and self.aug and np.random.rand() < 0.5:
            x, y = reverse_comp(x, y)
        return torch.from_numpy(x), torch.from_numpy(y)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, drop):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.bn = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(F.gelu(self.bn(self.conv(x))))


class Net(nn.Module):
    def __init__(self, hidden=256, drop=0.3, nblocks=3, kernel=5):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(14, 128, 1), nn.BatchNorm1d(128), nn.GELU())
        blocks = []
        cin = 128
        for i in range(nblocks):
            blocks.append(ConvBlock(cin, hidden, kernel, drop))
            cin = hidden
        self.blocks = nn.Sequential(*blocks)
        self.gru = nn.GRU(hidden, hidden // 2, num_layers=2, bidirectional=True,
                          batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(128, 5))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.inp(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        return self.head(x)


def mcrmse(pred, true, idx=SCORED_IDX):
    # pred,true: (B,L,5); score only first SCORED positions
    p = pred[:, :SCORED, :][..., idx]
    t = true[:, :SCORED, :][..., idx]
    se = ((p - t) ** 2).mean(dim=(0, 1))
    return torch.sqrt(se + 1e-8).mean()


def run_fold(fold, epochs, seed, outdir, lr, batch, use_all_targets=True,
             clip_aux=0.0, sn_only=False, aug=True, hidden=256, nblocks=3,
             drop=0.3, kernel=5, patience=15, select="sn1", tta=False, mixup=0.0,
             folds_file="blackboard/mirfak_zeta_cv_split.npy", asinh_s=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_items = load_jsonl("data/train.json")
    test_items = load_jsonl("data/test.json")
    folds = np.load(folds_file)
    tr = [it for it, f in zip(train_items, folds) if f != fold]
    va = [it for it, f in zip(train_items, folds) if f == fold]
    if sn_only:
        tr = [it for it in tr if it.get("SN_filter", 1) == 1]
    sn_mask = np.array([it.get("SN_filter", 1) == 1 for it in va])

    dl_tr = DataLoader(RNADataset(tr, train=True, use_all_targets=use_all_targets,
                                  clip_aux=clip_aux, aug=aug, asinh_s=asinh_s),
                       batch_size=batch, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(RNADataset(va, train=False, use_all_targets=use_all_targets,
                                  asinh_s=asinh_s),
                       batch_size=batch, shuffle=False, num_workers=4)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(hidden=hidden, drop=drop, nblocks=nblocks, kernel=kernel).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = 1e9
    best_state = None
    patience = 15
    bad = 0
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            if mixup > 0 and np.random.rand() < 0.5:
                lam = float(np.random.beta(mixup, mixup))
                perm = torch.randperm(x.size(0), device=dev)
                x = lam * x + (1 - lam) * x[perm]
                y = lam * y + (1 - lam) * y[perm]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                p = model(x)
                mask = torch.zeros(SEQ_LEN, device=dev)
                mask[:SCORED] = 1
                loss = (((p - y) ** 2).mean(-1) * mask).sum() / mask.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in dl_va:
                p = model(x.to(dev))
                preds.append(p.float().cpu())
                trues.append(y)
        pred = torch.cat(preds)
        true = torch.cat(trues)
        if asinh_s > 0:
            pred = torch.sinh(pred) * asinh_s
            true = torch.sinh(true) * asinh_s
        v = mcrmse(pred, true).item()
        if sn_mask.any():
            v_sn = mcrmse(pred[torch.from_numpy(sn_mask)],
                          true[torch.from_numpy(sn_mask)]).item()
        else:
            v_sn = v
        sel = v_sn if select == "sn1" else v
        if sel < best:
            best = sel
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"fold{fold} ep{ep} loss{tl/len(dl_tr):.4f} val{v:.4f} sn1{v_sn:.4f} best{best:.4f}", flush=True)
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    # OOF preds
    oof = np.zeros((len(va), SEQ_LEN, 5), dtype=np.float32)
    for i, it in enumerate(va):
        x = encode(it["sequence"], it["structure"], it["predicted_loop_type"])
        oof[i] = predict_batch(model, x[None], dev, tta)[0]
    # test preds
    tpred = np.zeros((len(test_items), SEQ_LEN, 5), dtype=np.float32)
    for i in range(0, len(test_items), batch):
        chunk = test_items[i:i + batch]
        xs = np.stack([encode(c["sequence"], c["structure"], c["predicted_loop_type"]) for c in chunk])
        tpred[i:i + len(chunk)] = predict_batch(model, xs, dev, tta)
    if asinh_s > 0:
        oof = np.sinh(oof) * asinh_s
        tpred = np.sinh(tpred) * asinh_s

    os.makedirs(outdir, exist_ok=True)
    np.save(f"{outdir}/oof_fold{fold}.npy", oof)
    np.save(f"{outdir}/va_idx_fold{fold}.npy",
            np.where(folds == fold)[0])
    np.save(f"{outdir}/test_fold{fold}.npy", tpred)
    with open(f"{outdir}/score_fold{fold}.txt", "w") as f:
        f.write(f"{best}\n")
    print(f"FOLD{fold} BEST {best:.4f}", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--outdir", default="workspaces/mirfak_zeta/preds")
    ap.add_argument("--scored_only", action="store_true")
    ap.add_argument("--clip_aux", type=float, default=0.0)
    ap.add_argument("--sn_only", action="store_true")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--nblocks", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--select", default="sn1", choices=["sn1", "all"])
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--asinh_s", type=float, default=0.0)
    ap.add_argument("--folds_file", default="blackboard/mirfak_zeta_cv_split.npy")
    a = ap.parse_args()
    t0 = time.time()
    run_fold(a.fold, a.epochs, a.seed, a.outdir, a.lr, a.batch,
             use_all_targets=not a.scored_only, clip_aux=a.clip_aux,
             sn_only=a.sn_only, aug=not a.no_aug, hidden=a.hidden,
             nblocks=a.nblocks, drop=a.drop, kernel=a.kernel, patience=a.patience,
             select=a.select, tta=a.tta, mixup=a.mixup, folds_file=a.folds_file,
             asinh_s=a.asinh_s)
    print(f"done in {time.time()-t0:.1f}s")
