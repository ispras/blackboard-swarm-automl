import os, time, argparse, json
import numpy as np, pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

p = argparse.ArgumentParser()
p.add_argument('--model', default='roberta-base')
p.add_argument('--tag', default=None)
p.add_argument('--maxlen', type=int, default=128)
p.add_argument('--epochs', type=int, default=4)
p.add_argument('--lr', type=float, default=2e-5)
p.add_argument('--bs', type=int, default=64)
p.add_argument('--folds', type=int, default=5)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--use_fast', type=int, default=1)
p.add_argument('--smooth', type=float, default=0.0)
p.add_argument('--ls', type=float, default=0.0)
args = p.parse_args()
tag = args.tag or args.model.replace('/', '_')
torch.manual_seed(args.seed); np.random.seed(args.seed)

ROOT = '/download_job_code/payload/sandbox'
tr = pd.read_csv(f'{ROOT}/data/train.csv'); te = pd.read_csv(f'{ROOT}/data/test.csv')
labels = ['EAP', 'HPL', 'MWS']; y = tr['author'].map({l: i for i, l in enumerate(labels)}).values

tok = AutoTokenizer.from_pretrained(args.model, use_fast=bool(args.use_fast))

class DS(Dataset):
    def __init__(self, texts, y=None):
        self.enc = tok(list(texts), truncation=True, max_length=args.maxlen, padding='max_length')
        self.y = y
    def __len__(self): return len(self.enc['input_ids'])
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.y is not None: d['labels'] = torch.tensor(self.y[i])
        return d

def infer(model, loader):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for b in loader:
            b = {k: v.cuda() for k, v in b.items() if k != 'labels'}
            logits = model(**b).logits.float()
            out.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(out)

te_ds = DS(te['text'].values); te_loader = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=2)

skf = StratifiedKFold(args.folds, shuffle=True, random_state=42)
oof = np.zeros((len(tr), 3)); test_pred = np.zeros((len(te), 3)); fold_scores = []
t0 = time.time()
for f, (ti, vi) in enumerate(skf.split(tr, y)):
    if f >= args.folds: break
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).cuda()
    tr_ds = DS(tr.iloc[ti]['text'].values, y[ti]); va_ds = DS(tr.iloc[vi]['text'].values, y[vi])
    tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(tr_loader) * args.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best = 9e9; best_state = None; best_ep = -1
    val_hist = []; te_hist = []
    for ep in range(args.epochs):
        model.train(); tot = 0; n = 0
        for b in tr_loader:
            b = {k: v.cuda() for k, v in b.items()}
            out = model(**{k:v for k,v in b.items() if k!='labels'})
            loss = F.cross_entropy(out.logits.float(), b['labels'], label_smoothing=args.ls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tot += loss.item() * len(b['input_ids']); n += len(b['input_ids'])
        vp = infer(model, va_loader); s = log_loss(y[vi], vp, labels=range(3))
        val_hist.append(vp.copy()); te_hist.append(infer(model, te_loader))
        print(f'[{tag}] fold{f} ep{ep} train={tot/n:.4f} val={s:.5f} {time.time()-t0:.0f}s', flush=True)
        if s < best:
            best = s; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # choose best tail of the epoch history (snapshot/tail-average)
    import itertools
    best_combo = None; E = len(val_hist)
    for r in range(1, E+1):
        for sub in itertools.combinations(range(E), r):
            avg = np.mean([val_hist[e] for e in sub], 0)
            sc = log_loss(y[vi], avg, labels=range(3))
            if best_combo is None or sc < best_combo[0]:
                best_combo = (sc, sub)
    sc, sub = best_combo
    model.load_state_dict(best_state)
    oof[vi] = np.mean([val_hist[e] for e in sub], 0)
    test_pred += np.mean([te_hist[e] for e in sub], 0) / args.folds
    fold_scores.append(sc)
    print(f'[{tag}] fold{f} SNAP {sub} val={sc:.5f} (best-single {best:.5f} ep{best_ep})', flush=True)
    del model; torch.cuda.empty_cache()

oof_s = log_loss(y, oof, labels=range(3))
print(f'[{tag}] OOF {oof_s:.6f} folds={[round(x,5) for x in fold_scores]}', flush=True)
np.save(f'oof_{tag}.npy', oof); np.save(f'test_{tag}.npy', test_pred)
sub = pd.DataFrame({'id': te['id'], 'EAP': test_pred[:, 0], 'HPL': test_pred[:, 1], 'MWS': test_pred[:, 2]})
sub.to_csv(f'sub_{tag}.csv', index=False)
json.dump({'tag': tag, 'oof': oof_s, 'folds': fold_scores, 'args': vars(args)}, open(f'meta_{tag}.json', 'w'), indent=2)
