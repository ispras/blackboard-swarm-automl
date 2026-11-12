import os, time, argparse, json, gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from common import load_data, folds, logloss, WS, CLASSES

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len,
                       padding=False, return_length=False)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        n = maxlen - len(b["input_ids"])
        out["input_ids"].append(torch.cat([b["input_ids"], torch.full((n,), pad_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(n, dtype=torch.long)]))
        out["labels"].append(b["labels"])
    for k in out:
        out[k] = torch.stack(out[k])
    return out


AMP = torch.bfloat16


class FGM:
    def __init__(self, model, eps=1.0):
        self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and \
                    ("embeddings" in name or "word_emb" in name):
                norm = torch.norm(param.grad)
                if norm == 0 or torch.isnan(norm):
                    continue
                self.backup[name] = param.data.clone()
                param.data.add_(self.eps * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(dtype=AMP):
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
        ps.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(ps, 0)


def train_one(train_txt, train_y, val_txt, val_y, test_txt, model_name, args, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, problem_type="single_label_classification").to(device)

    tr_ds = DS(train_txt, train_y, tok, args.max_len)
    va_loader = DataLoader(DS(val_txt, val_y, tok, args.max_len), batch_size=args.bs * 2,
                           shuffle=False, collate_fn=lambda b: collate(b, tok.pad_token_id or 0))
    te_loader = DataLoader(DS(test_txt, [0] * len(test_txt), tok, args.max_len), batch_size=args.bs * 2,
                           shuffle=False, collate_fn=lambda b: collate(b, tok.pad_token_id or 0))
    tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                           collate_fn=lambda b: collate(b, tok.pad_token_id or 0))

    n_steps = len(tr_loader) * args.epochs // args.grad_accum
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * n_steps), n_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.ls)
    fgm = FGM(model, eps=args.fgm) if args.fgm > 0 else None

    best_ll, best_va, best_te, best_ep = 1e9, None, None, -1
    va_hist, te_hist = [], []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        opt.zero_grad()
        for i, batch in enumerate(tr_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(dtype=AMP):
                logits = model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
                loss = loss_fn(logits, batch["labels"]) / args.grad_accum
            loss.backward()
            if fgm is not None:
                fgm.attack()
                with autocast(dtype=AMP):
                    logits_adv = model(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"]).logits
                    loss_adv = loss_fn(logits_adv, batch["labels"]) / args.grad_accum
                loss_adv.backward()
                fgm.restore()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        va_p = predict(model, va_loader, device)
        te_p = predict(model, te_loader, device)
        ll = logloss(val_y, va_p)
        acc = (va_p.argmax(1) == val_y).mean()
        print(f"  ep{ep} val_ll={ll:.5f} val_acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        va_hist.append(va_p); te_hist.append(te_p)
        if ll < best_ll:
            best_ll, best_va, best_te, best_ep = ll, va_p, te_p, ep
    k = max(1, args.epochs // 2)
    va_avg = np.mean(va_hist[-k:], 0)
    te_avg = np.mean(te_hist[-k:], 0)
    return best_ll, best_va, best_te, best_ep, va_avg, te_avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--tag", default="deberta")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--fgm", type=float, default=0.0)
    ap.add_argument("--nfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mseed", type=int, default=42)
    ap.add_argument("--folds_only", type=str, default="")
    ap.add_argument("--all_data", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.mseed)
    np.random.seed(args.mseed)
    tr, te, y = load_data()
    test_txt = te["text"].values

    if args.all_data:
        res = train_one(tr["text"].values, y, tr["text"].values[:200], y[:200],
                        test_txt, args.model, args, device)
        np.save(os.path.join(WS, f"test_{args.tag}_alldata.npy"), res[2])
        np.save(os.path.join(WS, f"test_{args.tag}_alldata_epavg.npy"), res[5])
        print("saved alldata test probs")
        return

    flist = folds(y, n=args.nfold, seed=args.seed)
    oof = np.zeros((len(y), 3))
    test_pred = np.zeros((len(test_txt), 3))
    oof_avg = np.zeros((len(y), 3))
    test_avg = np.zeros((len(test_txt), 3))
    fold_ids = list(range(args.nfold))
    if args.folds_only:
        fold_ids = [int(x) for x in args.folds_only.split(",")]
    for f in fold_ids:
        itr, iva = flist[f]
        print(f"=== fold {f} ===", flush=True)
        res = train_one(
            tr["text"].values[itr], y[itr], tr["text"].values[iva], y[iva],
            test_txt, args.model, args, device)
        best_ll, va_p, te_p, ep, ava_p, ate_p = res
        oof[iva] = va_p
        test_pred += te_p / len(fold_ids)
        oof_avg[iva] = ava_p
        test_avg += ate_p / len(fold_ids)
        print(f"fold {f} best ep{ep} ll={best_ll:.5f} epavg_ll={logloss(y[iva], ava_p):.5f}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    ll = logloss(y, oof)
    print(f"OOF logloss = {ll:.5f}  (epavg {logloss(y, oof_avg):.5f})")
    np.save(os.path.join(WS, f"oof_{args.tag}.npy"), oof)
    np.save(os.path.join(WS, f"test_{args.tag}.npy"), test_pred)
    np.save(os.path.join(WS, f"oof_{args.tag}_epavg.npy"), oof_avg)
    np.save(os.path.join(WS, f"test_{args.tag}_epavg.npy"), test_avg)
    print("saved", f"oof_{args.tag}.npy")


if __name__ == "__main__":
    main()
