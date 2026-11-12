import argparse, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import log_loss

WS = "workspaces/arcturus_alpha"
CLASSES = ["EAP", "HPL", "MWS"]


class DS(Dataset):
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
    model.eval(); outs = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask"))
        outs.append(torch.softmax(out.logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(outs, 0)


def run_fold(model_name, tr, va, te_texts, fold, args, device):
    torch.manual_seed(args.seed * 1000 + fold)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
    for attr in ["hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout", "dropout", "summary_last_dropout"]:
        if hasattr(model.config, attr):
            setattr(model.config, attr, args.dropout)
    tr_dl = DataLoader(DS(tr["text"], tr["y"].values, tok, args.max_len), batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    va_dl = DataLoader(DS(va["text"], va["y"].values, tok, args.max_len), batch_size=64, shuffle=False, num_workers=2)
    te_dl = DataLoader(DS(te_texts, None, tok, args.max_len), batch_size=64, shuffle=False, num_workers=2)
    no_decay = ["bias", "LayerNorm.weight"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.wd},
              {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=args.lr)
    total = len(tr_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    crit = nn.CrossEntropyLoss(label_smoothing=args.ls)
    emb = model.get_input_embeddings()
    best, best_va, best_te = 1e9, None, None
    avg_va = np.zeros((len(va), 3)); avg_te = np.zeros((len(te_texts), 3))
    for ep in range(args.epochs):
        model.train(); rl = 0.0; nb = 0; t0 = time.time()
        for b in tr_dl:
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask"))
                loss = crit(out.logits, b["labels"])
            loss.backward()
            if args.fgm > 0:
                g = emb.weight.grad
                if g is not None:
                    norm = g.norm()
                    if norm > 0:
                        r = args.fgm * g / (norm + 1e-8)
                        emb.weight.data.add_(r)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                            out2 = model(input_ids=b["input_ids"], attention_mask=b.get("attention_mask"))
                            loss2 = crit(out2.logits, b["labels"])
                        loss2.backward()
                        emb.weight.data.sub_(r)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); rl += loss.item(); nb += 1
        vp = predict(model, va_dl, device); tp = predict(model, te_dl, device)
        avg_va += vp / args.epochs; avg_te += tp / args.epochs
        ll = log_loss(va["author"].values, vp, labels=CLASSES)
        print(f"  fold {fold} ep {ep} loss {rl/nb:.4f} val {ll:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if ll < best:
            best, best_va, best_te = ll, vp, tp
    return best, best_va, best_te, avg_va, avg_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlnet-base-cased")
    ap.add_argument("--tag", default="xlnet")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--ls", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--fgm", type=float, default=0.0)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
    train["y"] = train["author"].map({c: i for i, c in enumerate(CLASSES)})
    train["fold"] = np.load(f"{WS}/folds.npy")
    oof = np.zeros((len(train), 3)); test_p = np.zeros((len(test), 3))
    oof_a = np.zeros((len(train), 3)); test_a = np.zeros((len(test), 3))
    fl = [int(x) for x in args.folds.split(",")]
    for f in fl:
        print(f"=== {args.tag} fold {f} ===", flush=True)
        tr = train[train.fold != f]; va = train[train.fold == f]
        best, vp, tp, avp, atp = run_fold(args.model, tr, va, test["text"].values, f, args, device)
        m = train.fold.values == f
        oof[m] = vp; test_p += tp / len(fl); oof_a[m] = avp; test_a += atp / len(fl)
        np.save(f"{WS}/oof_{args.tag}.npy", oof); np.save(f"{WS}/test_{args.tag}.npy", test_p)
        np.save(f"{WS}/oof_{args.tag}_epavg.npy", oof_a); np.save(f"{WS}/test_{args.tag}_epavg.npy", test_a)
        print(f"BEST fold {f}: {best:.5f}", flush=True)
    m = np.isin(train.fold.values, fl)
    print(f"OOF {args.tag} best {log_loss(train.author.values[m], oof[m], labels=CLASSES):.5f} "
          f"avg {log_loss(train.author.values[m], oof_a[m], labels=CLASSES):.5f}")
    np.save(f"{WS}/oof_{args.tag}.npy", oof); np.save(f"{WS}/test_{args.tag}.npy", test_p)
    np.save(f"{WS}/oof_{args.tag}_epavg.npy", oof_a); np.save(f"{WS}/test_{args.tag}_epavg.npy", test_a)


if __name__ == "__main__":
    main()
