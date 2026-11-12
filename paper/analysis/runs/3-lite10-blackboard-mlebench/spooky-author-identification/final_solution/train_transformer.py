import os, sys, json, time, argparse, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          set_seed)
from transformers.trainer_utils import EvalPrediction

ROOT = '/download_job_code/payload/sandbox'
CLASSES = ['EAP', 'HPL', 'MWS']
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--epochs', type=float, default=4.0)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--max_len', type=int, default=160)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup', type=float, default=0.1)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--only_fold', type=int, default=-1)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--trust_remote_code', action='store_true')
    return p.parse_args()

class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            d['labels'] = self.labels[i]
        return d

def main():
    a = parse()
    set_seed(a.seed)
    os.makedirs(f'{ROOT}/workspaces/aldebaran_xi/preds', exist_ok=True)
    os.makedirs(f'{ROOT}/workspaces/aldebaran_xi/logs', exist_ok=True)

    tr = pd.read_csv(f'{ROOT}/data/train.csv')
    te = pd.read_csv(f'{ROOT}/data/test.csv')
    y = tr['author'].map(LABEL2ID).values
    texts = tr['text'].values
    test_texts = te['text'].values

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    model_cls = AutoModelForSequenceClassification

    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=42)
    splits = list(skf.split(texts, y))

    oof = np.zeros((len(tr), 3))
    test_pred = np.zeros((len(te), 3))
    fold_scores = []

    logf = open(f'{ROOT}/workspaces/aldebaran_xi/logs/{a.tag}.log', 'a')
    def log(*x):
        s = ' '.join(str(v) for v in x)
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    log(f'== {a.tag} model={a.model} args={vars(a)}')
    for fold, (tr_idx, va_idx) in enumerate(splits):
        if a.only_fold >= 0 and fold != a.only_fold:
            continue
        t0 = time.time()
        tr_ds = DS(texts[tr_idx], y[tr_idx], tok, a.max_len)
        va_ds = DS(texts[va_idx], y[va_idx], tok, a.max_len)
        te_ds = DS(test_texts, [0]*len(test_texts), tok, a.max_len)

        model = model_cls.from_pretrained(a.model, num_labels=3,
                                          trust_remote_code=a.trust_remote_code)
        args = TrainingArguments(
            output_dir=f'{ROOT}/workspaces/aldebaran_xi/ckpt/{a.tag}_f{fold}',
            num_train_epochs=a.epochs, learning_rate=a.lr,
            per_device_train_batch_size=a.bs, per_device_eval_batch_size=64,
            gradient_accumulation_steps=a.grad_accum,
            warmup_ratio=a.warmup, weight_decay=a.wd,
            eval_strategy='epoch', save_strategy='epoch',
            load_best_model_at_end=True, metric_for_best_model='eval_loss',
            greater_is_better=False, save_total_limit=1,
            logging_steps=100, bf16=True, report_to='none',
            seed=a.seed, data_seed=a.seed, dataloader_num_workers=2,
        )
        coll = DataCollatorWithPadding(tok)
        trainer = Trainer(model=model, args=args, train_dataset=tr_ds,
                          eval_dataset=va_ds, tokenizer=tok, data_collator=coll)
        trainer.train()
        pv = trainer.predict(va_ds)
        logits = pv.predictions
        prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        oof[va_idx] = prob
        sc = log_loss(y[va_idx], prob, labels=[0,1,2])
        pt = trainer.predict(te_ds)
        test_pred += torch.softmax(torch.tensor(pt.predictions), dim=-1).numpy() / (a.folds if a.only_fold < 0 else 1)
        fold_scores.append(sc)
        log(f'fold {fold}: logloss={sc:.5f} time={time.time()-t0:.0f}s')
        del model, trainer
        torch.cuda.empty_cache()

    if a.only_fold < 0:
        overall = log_loss(y, np.clip(oof, 1e-15, 1), labels=[0,1,2])
        log(f'OVERALL OOF logloss={overall:.5f} folds={fold_scores}')
        np.save(f'{ROOT}/workspaces/aldebaran_xi/preds/oof_{a.tag}.npy', oof)
        np.save(f'{ROOT}/workspaces/aldebaran_xi/preds/test_{a.tag}.npy', test_pred)
        json.dump({'tag': a.tag, 'model': a.model, 'oof_logloss': overall,
                   'fold_scores': fold_scores, 'args': vars(a)},
                  open(f'{ROOT}/workspaces/aldebaran_xi/preds/meta_{a.tag}.json','w'), indent=2)
    logf.close()

if __name__ == '__main__':
    main()
