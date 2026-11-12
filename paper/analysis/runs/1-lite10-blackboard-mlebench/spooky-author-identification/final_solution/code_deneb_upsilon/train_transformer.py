import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import log_loss

CLASSES = ["EAP", "HPL", "MWS"]


class SpookyDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts = list(texts)
        self.labels = labels
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def run_fold(model_name, train_df, test_texts, fold, args, device):
    torch.manual_seed(args.seed * 1000 + fold)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    if args.dropout > 0:
        for attr in ["hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout", "dropout"]:
            if hasattr(model.config, attr):
                setattr(model.config, attr, args.dropout)
    model = model.to(device)
    if args.freeze > 0:
        base = getattr(model, "roberta", None) or getattr(model, "bert", None) or getattr(model, "deberta", None)
        if base is not None:
            for p in base.embeddings.parameters():
                p.requires_grad = False
            layers = base.encoder.layer
            for i, layer in enumerate(layers):
                if i < args.freeze:
                    for p in layer.parameters():
                        p.requires_grad = False
            print(f"  frozen embeddings + {args.freeze}/{len(layers)} layers", flush=True)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    tr = train_df[train_df["fold"] != fold].reset_index(drop=True)
    va = train_df[train_df["fold"] == fold].reset_index(drop=True)
    ymap = {c: i for i, c in enumerate(CLASSES)}
    tr_ds = SpookyDS(tr["text"], [ymap[a] for a in tr["author"]], tok, args.max_len)
    va_ds = SpookyDS(va["text"], [ymap[a] for a in va["author"]], tok, args.max_len)
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=2)
    te_ds = SpookyDS(test_texts, None, tok, args.max_len)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=2)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.wd},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    total = len(tr_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=args.ls)

    best = 1e9
    best_va = None
    best_te = None
    avg_va = 0.0
    avg_te = 0.0
    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        nb = 0
        for batch in tr_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = crit(out.logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            nb += 1
        print(f"  fold {fold} ep {ep} train_loss {run_loss/nb:.4f} gn {gn.item():.3f} lr {sched.get_last_lr()[0]:.2e}", flush=True)

        model.eval()
        vp = predict(model, va_dl, device)
        tp = predict(model, te_dl, device)
        avg_va = avg_va + vp
        avg_te = avg_te + tp
        ll = log_loss(va["author"].values, vp, labels=CLASSES)
        print(f"  fold {fold} ep {ep} val_logloss {ll:.5f}", flush=True)
        if ll < best:
            best = ll
            best_va = vp
            best_te = tp
    return best, best_va, best_te, avg_va / args.epochs, avg_te / args.epochs


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    outs = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        outs.append(torch.softmax(out.logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--tag", default="roberta")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    folds = pd.read_csv("workspaces/deneb_upsilon/shared_folds.csv")[["id", "fold"]]
    train = train.merge(folds, on="id", how="left")
    ymap = {c: i for i, c in enumerate(CLASSES)}
    train["y"] = train["author"].map(ymap)

    oof = np.zeros((len(train), 3))
    test_pred = np.zeros((len(test), 3))
    oof_avg = np.zeros((len(train), 3))
    test_avg = np.zeros((len(test), 3))
    fold_list = [int(x) for x in args.folds.split(",")]
    for f in fold_list:
        print(f"=== {args.tag} fold {f} ===", flush=True)
        best, va, te, vaa, tea = run_fold(args.model, train, test["text"].values, f, args, device)
        mask = train["fold"].values == f
        oof[mask] = va
        test_pred += te / len(fold_list)
        oof_avg[mask] = vaa
        test_avg += tea / len(fold_list)
        print(f"BEST fold {f}: {best:.5f}", flush=True)

    mask = np.isin(train["fold"].values, fold_list)
    total = log_loss(train["author"].values[mask], oof[mask], labels=CLASSES)
    total_avg = log_loss(train["author"].values[mask], oof_avg[mask], labels=CLASSES)
    print(f"OOF {args.tag} ({len(fold_list)} folds) LOGLOSS {total:.5f} | AVG {total_avg:.5f}")
    np.save(f"workspaces/deneb_upsilon/{args.tag}_oof.npy", oof)
    np.save(f"workspaces/deneb_upsilon/{args.tag}_test.npy", test_pred)
    np.save(f"workspaces/deneb_upsilon/{args.tag}avg_oof.npy", oof_avg)
    np.save(f"workspaces/deneb_upsilon/{args.tag}avg_test.npy", test_avg)
    sub = pd.DataFrame({"id": test["id"], "EAP": test_pred[:, 0], "HPL": test_pred[:, 1],
                        "MWS": test_pred[:, 2]})
    sub.to_csv(f"workspaces/deneb_upsilon/submission_{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
