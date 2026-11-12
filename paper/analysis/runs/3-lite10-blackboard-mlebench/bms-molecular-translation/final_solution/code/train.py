import sys, os, time, math, json, argparse
sys.path.append('/usr/local/lib/python3.11/site-packages')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from tokenizers import Tokenizer
from rapidfuzz.distance import Levenshtein
from model import Model

cv2.setNumThreads(0)
ROOT = '/download_job_code/payload/sandbox'
WORK = f'{ROOT}/workspaces/rigel_mu'

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DS(Dataset):
    def __init__(self, rows, img_size, tokens, seqlens):
        self.rows = rows
        self.img_size = img_size
        self.tokens = tokens
        self.seqlens = seqlens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        iid = self.ids[r]
        p = f'{ROOT}/data/train/{iid[0]}/{iid[1]}/{iid[2]}/{iid}.png'
        img = cv2.imread(p)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))
        L = int(self.seqlens[r])
        tok = np.asarray(self.tokens[r, :L], dtype=np.int64)
        return torch.from_numpy(img), torch.from_numpy(tok)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    T = max(b[1].numel() for b in batch)
    out = torch.zeros(len(batch), T, dtype=torch.long)
    for i, b in enumerate(batch):
        out[i, :b[1].numel()] = b[1]
    return imgs, out


def lev_metric(preds, gts):
    return float(np.mean([Levenshtein.distance(p, g) for p, g in zip(preds, gts)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_size', type=int, default=384)
    ap.add_argument('--batch', type=int, default=96)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--enc_lr_mult', type=float, default=0.3)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--epochs', type=float, default=100)
    ap.add_argument('--max_steps', type=int, default=100000000)
    ap.add_argument('--n_train', type=int, default=0, help='0 = all train rows')
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--eval_n', type=int, default=1024)
    ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--dim_ff', type=int, default=2048)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--enc', default='convnext_tiny.fb_in22k_ft_in1k')
    ap.add_argument('--out_index', type=int, default=2)
    ap.add_argument('--resume', default='')
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--bench', type=int, default=0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)

    tok = Tokenizer.from_file(f'{WORK}/bpe1024.json')
    PAD, BOS, EOS = tok.token_to_id('[PAD]'), tok.token_to_id('[BOS]'), tok.token_to_id('[EOS]')
    vocab = tok.get_vocab_size()

    meta = pd.read_csv(f'{WORK}/labels_meta.csv')
    ids = meta.image_id.values
    tokens = np.load(f'{WORK}/labels_tokens.npy', mmap_mode='r')
    seqlens = np.load(f'{WORK}/labels_seqlens.npy')
    is_val = np.load(f'{WORK}/is_val.npy')

    if os.path.exists(f'{WORK}/train_order.npy'):
        order = np.load(f'{WORK}/train_order.npy')
    else:
        tr = np.where(~is_val)[0]
        rr = np.random.RandomState(123).permutation(len(tr))
        order = tr[rr]
        np.save(f'{WORK}/train_order.npy', order)
    if args.n_train > 0:
        order = order[:args.n_train]

    val_rows = np.where(is_val)[0]
    # deterministic subset for eval
    eval_rows = np.random.RandomState(7).permutation(val_rows)[:args.eval_n]

    DS.ids = ids
    DS.tokens = tokens
    DS.seqlens = seqlens
    train_ds = DS(order, args.img_size, tokens, seqlens)
    eval_ds = DS(eval_rows, args.img_size, tokens, seqlens)

    model = Model(vocab, enc_name=args.enc, out_index=args.out_index, d_model=args.d_model,
                  nhead=args.nhead, layers=args.layers, dim_ff=args.dim_ff,
                  dropout=args.dropout, max_len=tokens.shape[1]).cuda()
    nparams = sum(p.numel() for p in model.parameters())
    print(f'params {nparams/1e6:.1f}M', flush=True)

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW([
        {'params': [p for p in decay if id(p) not in enc_ids], 'weight_decay': args.wd},
        {'params': [p for p in no_decay if id(p) not in enc_ids], 'weight_decay': 0.0},
        {'params': [p for p in decay if id(p) in enc_ids], 'weight_decay': args.wd, 'lr': args.lr * args.enc_lr_mult},
        {'params': [p for p in no_decay if id(p) in enc_ids], 'weight_decay': 0.0, 'lr': args.lr * args.enc_lr_mult},
    ], lr=args.lr, betas=(0.9, 0.98), eps=1e-6)

    last_epoch = 1.0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        global_step = ck['step']
        last_epoch = ck.get('last_epoch', 1.0)
        print('resumed from', args.resume, 'step', global_step, flush=True)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch)
    total_steps = min(args.max_steps, int(steps_per_epoch * args.epochs))

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, pin_memory=True, drop_last=True,
                    persistent_workers=True, prefetch_factor=4)

    logf = open(f'{WORK}/train_{args.tag}.log', 'a')

    def log(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log('START', json.dumps(vars(args)))
    log('steps_per_epoch', steps_per_epoch, 'total_steps', total_steps)

    @torch.no_grad()
    def evaluate():
        model.eval()
        edl = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=8,
                         collate_fn=collate, pin_memory=True)
        preds, gts = [], []
        for imgs, toks in edl:
            imgs = imgs.cuda(non_blocking=True)
            outs = model.generate(imgs, BOS, EOS, max_len=min(320, tokens.shape[1]))
            for o in outs:
                preds.append(tok.decode(o))
            for t in range(toks.shape[0]):
                L = int(seqlens[eval_rows[len(gts)]])
                gts.append(tok.decode(toks[t, 1:L-1].tolist()))
            if len(preds) >= args.eval_n:
                break
        sc = lev_metric(preds, gts)
        model.train()
        return sc

    if args.bench:
        imgs, toks = next(iter(dl))
        imgs = imgs.cuda(non_blocking=True)
        tin = toks[:, :-1].cuda()
        tout = toks[:, 1:].cuda()
        for _ in range(3):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(imgs, tin)
                loss = F.cross_entropy(logits.reshape(-1, vocab), tout.reshape(-1),
                                       ignore_index=PAD, label_smoothing=0.05)
        torch.cuda.synchronize()
        t0 = time.time(); n = args.bench
        for _ in range(n):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(imgs, tin)
                loss = F.cross_entropy(logits.reshape(-1, vocab), tout.reshape(-1),
                                       ignore_index=PAD, label_smoothing=0.05)
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        log(f'BENCH {n} steps {dt:.2f}s -> {n*args.batch/dt:.1f} img/s, loss {loss.item():.4f}')
        return

    model.train()
    best = 1e9
    t_start = time.time()
    step = global_step
    for ep in range(int(math.ceil(args.epochs))):
        for imgs, toks in dl:
            if step >= total_steps:
                break
            lf = lr_at(step)
            for gi, g in enumerate(opt.param_groups):
                g['lr'] = args.lr * lf * (args.enc_lr_mult if gi >= 2 else 1.0)
            imgs = imgs.cuda(non_blocking=True)
            tin = toks[:, :-1].cuda()
            tout = toks[:, 1:].cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(imgs, tin)
                loss = F.cross_entropy(logits.reshape(-1, vocab), tout.reshape(-1),
                                       ignore_index=PAD, label_smoothing=0.05)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                el = time.time() - t_start
                ips = (step - global_step) * args.batch / max(1e-9, el)
                log(f'step {step} ep {ep} loss {loss.item():.4f} gn {gn.item():.2f} '
                    f'lr {opt.param_groups[0]["lr"]:.2e} {ips:.0f}img/s eta_h {el/(step-global_step)*(total_steps-step)/3600:.2f}')
            if step % args.save_every == 0 or step == total_steps:
                torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                            'step': step, 'args': vars(args)}, f'{WORK}/ck_{args.tag}_last.pt')
            if step % args.eval_every == 0 or step == total_steps:
                sc = evaluate()
                log(f'EVAL step {step} lev {sc:.3f}')
                if sc < best:
                    best = sc
                    torch.save({'model': model.state_dict(), 'step': step, 'lev': sc,
                                'args': vars(args)}, f'{WORK}/ck_{args.tag}_best.pt')
        if step >= total_steps:
            break
    log('DONE best', best)


if __name__ == '__main__':
    main()
