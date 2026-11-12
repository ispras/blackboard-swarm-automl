import os, sys, time, argparse, pickle, csv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def build_vocab(tok):
    V = tok.vocab_size; ids = []; id2w = {}
    for i in range(V):
        s = tok.convert_ids_to_tokens(i)
        if s is None or s.startswith('<') or s.startswith('Ġ<'): continue
        if not s.startswith('Ġ'): continue
        w = s[1:]
        if w == '' or '\\u' in repr(w): continue
        id2w[i] = w; ids.append(i)
    return np.array(ids, dtype=np.int64), id2w

def load_input(path, kind):
    if kind == 'csv':
        sents = []
        with open(path, newline='') as f:
            rd = csv.reader(f); next(rd)
            for row in rd:
                if len(row) >= 2: sents.append(row[1])
        return sents
    return [l.rstrip('\n') for l in open(path)]

@torch.inference_mode()
def dump(args):
    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    dt = torch.float16 if (args.fp16 and dev == 'cuda') else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(args.ckpt, torch_dtype=dt).eval().to(dev)
    ids_all, id2w = build_vocab(tok)
    ids_t = torch.tensor(ids_all, device=dev)
    sents = load_input(args.input, args.kind)
    if args.limit: sents = sents[:args.limit]
    toks_list = [s.split() for s in sents]
    print('sentences', len(toks_list), 'allowed', len(ids_all), flush=True)
    meta = []; texts = []
    for si, t in enumerate(toks_list):
        for g in range(1, len(t)):
            texts.append(' '.join(t[:g] + ['<mask>'] + t[g:])); meta.append((si, g))
    print('variants', len(texts), flush=True)
    out = [dict() for _ in toks_list]
    mask_id = tok.mask_token_id
    t0 = time.time()
    for s in range(0, len(texts), args.bs):
        chunk = texts[s:s+args.bs]
        enc = tok(chunk, padding='longest', truncation=True, max_length=args.maxlen, return_tensors='pt')
        ii = enc['input_ids'].to(dev, non_blocking=True); am = enc['attention_mask'].to(dev, non_blocking=True)
        lo = model(input_ids=ii, attention_mask=am).logits
        mp = (ii == mask_id); has = mp.any(1)
        idx = mp.float().argmax(1)
        ar = torch.arange(len(chunk), device=dev)
        lg = lo[ar, idx].float()
        pr = torch.softmax(lg, -1)[:, ids_t]
        k = min(args.topk, pr.size(1))
        tv, ti = pr.topk(k, -1)
        tv = tv.half().cpu().numpy(); ti = ti.cpu().numpy()
        has_np = has.cpu().numpy()
        for r in range(len(chunk)):
            if not has_np[r]: continue
            si, g = meta[s + r]
            ws = [id2w[int(ids_all[ti[r, j]])] for j in range(k)]
            ps = [float(tv[r, j]) for j in range(k)]
            out[si][g] = {'w': ws, 'p': ps, 'maxp': ps[0] if ps else 0.0}
        if s % (args.bs * 40) == 0:
            el = time.time() - t0
            print(f'{s}/{len(texts)} {len(chunk)} { (s+len(chunk))/max(el,1):.0f}/s elapsed {el:.0f}s', flush=True)
    pickle.dump(out, open(args.out, 'wb'), protocol=4)
    print('saved', args.out, round(time.time() - t0, 1), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--kind', default='txt', choices=['txt', 'csv'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--maxlen', type=int, default=192)
    ap.add_argument('--fp16', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dump(args)

if __name__ == '__main__':
    main()
