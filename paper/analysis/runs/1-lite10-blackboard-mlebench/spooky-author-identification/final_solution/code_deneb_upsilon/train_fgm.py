import argparse
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
        self.texts = list(texts); self.labels = labels; self.tok = tok; self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    outs = []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        outs.append(torch.softmax(out.logits.float(), -1).cpu().numpy())
    return np.concatenate(outs, 0)


def run_fold(model_name, train_df, test_texts, fold, args, device):
    torch.manual_seed(args.seed * 1000 + fold)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
    tr = train_df[train_df["fold"] != fold].reset_index(drop=True)
    va = train_df[train_df["fold"] == fold].reset_index(drop=True)
    ymap = {c: i for i, c in enumerate(CLASSES)}
    tr_dl = DataLoader(SpookyDS(tr["text"], [ymap[a] for a in tr["author"]], tok, args.max_len),
                       batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    va_dl = DataLoader(SpookyDS(va["text"], [ymap[a] for a in va["author"]], tok, args.max_len),
                       batch_size=64, num_workers=2)
    te_dl = DataLoader(SpookyDS(test_texts, None, tok, args.max_len), batch_size=64, num_workers=2)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.wd},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    total = len(tr_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    crit = nn.CrossEntropyLoss(label_smoothing=args.ls)
    emb = model.get_input_embeddings()

    best = 1e9; best_va = best_te = None
    avg_va = avg_te = 0.0
    for ep in range(args.epochs):
        model.train()
        for batch in tr_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                enabled=(device == "cuda")):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = crit(out.logits, batch["labels"])
            loss.backward()
            if args.fgm > 0:
                g = emb.weight.grad
                if g is not None:
                    norm = g.norm()
                    if norm > 0:
                        r = args.fgm * g / (norm + 1e-8)
                        emb.weight.data.add_(r)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                            out2 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                            loss2 = crit(out2.logits, batch["labels"])
                        loss2.backward()
                        emb.weight.data.sub_(r)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        vp = predict(model, va_dl, device); tp = predict(model, te_dl, device)
        avg_va = avg_va + vp; avg_te = avg_te + tp
        ll = log_loss(va["author"].values, vp, labels=CLASSES)
        print(f"  fold {fold} ep {ep} val {ll:.5f}", flush=True)
        if ll < best:
            best = ll; best_va = vp; best_te = tp
    return best, best_va, best_te, avg_va / args.epochs, avg_te / args.epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--tag", default="fgm_bertcased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--ls", type=float, default=0.05)
    ap.add_argument("--fgm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train = train.merge(pd.read_csv("workspaces/deneb_upsilon/shared_folds.csv")[["id", "fold"]], on="id", how="left")
    oof = np.zeros((len(train), 3)); oof_avg = np.zeros((len(train), 3))
    tp = np.zeros((len(test), 3)); ta = np.zeros((len(test), 3))
    fl = [int(x) for x in args.folds.split(",")]
    for f in fl:
        print(f"=== {args.tag} fold {f} ===", flush=True)
        best, va, te, vaa, tea = run_fold(args.model, train, test["text"].values, f, args, device)
        m = train["fold"].values == f
        oof[m] = va; tp += te / len(fl); oof_avg[m] = vaa; ta += tea / len(fl)
        print(f"BEST fold {f}: {best:.5f}", flush=True)
    m = np.isin(train["fold"].values, fl)
    print(f"OOF {args.tag} LOGLOSS {log_loss(train['author'].values[m], oof[m], labels=CLASSES):.5f} | AVG {log_loss(train['author'].values[m], oof_avg[m], labels=CLASSES):.5f}")
    for name, arr in [(f"{args.tag}_oof", oof), (f"{args.tag}_test", tp),
                      (f"{args.tag}avg_oof", oof_avg), (f"{args.tag}avg_test", ta)]:
        np.save(f"workspaces/deneb_upsilon/{name}.npy", arr)
    pd.DataFrame({"id": test["id"], "EAP": tp[:, 0], "HPL": tp[:, 1], "MWS": tp[:, 2]}).to_csv(
        f"workspaces/deneb_upsilon/submission_{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
