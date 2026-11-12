"""roberta-base MLM imputation scores on the shared val holdout.

For every candidate gap g=1..n-1 of the deleted sentence T, insert [MASK] between
T[g-1] and T[g] and record top-k conditional P(w|context,g).

Output JSONL: {"id": int, "pred": {str(g): {"w":[...], "lp":[...]}}}
"""
import argparse, json, csv, sys, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def norm_tok(t):
    return t.replace('\u0120', ' ').replace('\u010a', ' ').replace('\u0109', ' ').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='roberta-base')
    ap.add_argument('--test', default='val_test.csv')
    ap.add_argument('--out', default='mlm_val20k.jsonl')
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--maxlen', type=int, default=256)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.test, encoding='utf-8')))[:args.n]
    tok = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.eval().cuda()
    mask_id = tok.mask_token_id
    dev = 'cuda'

    items = []
    for di, r in enumerate(rows):
        words = [w for w in r['sentence'].split(' ') if w != '']
        enc = tok(words, is_split_into_words=True, add_special_tokens=True,
                  truncation=True, max_length=args.maxlen)
        ids = enc['input_ids']; wids = enc.word_ids()
        first_tok = {}
        for ti, w in enumerate(wids):
            if w is not None and w not in first_tok:
                first_tok[w] = ti
        nw = len(words)
        for g in range(1, nw):
            if g not in first_tok:
                continue
            items.append((di, g, ids, first_tok[g]))
    print('items', len(items), 'docs', len(rows), flush=True)

    results = [dict() for _ in range(len(rows))]
    vocab = model.config.vocab_size
    with torch.no_grad():
        for bi in range(0, len(items), args.batch):
            batch = items[bi:bi+args.batch]
            maxlen = max(len(x[2]) + 1 for x in batch)
            inp = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(batch), maxlen), dtype=torch.long)
            posl = []
            for j, (di, g, ids, pos) in enumerate(batch):
                seq = ids[:pos] + [mask_id] + ids[pos:]
                L = len(seq)
                inp[j, :L] = torch.tensor(seq, dtype=torch.long)
                att[j, :L] = 1
                posl.append(pos)
            inp = inp.to(dev); att = att.to(dev)
            logits = model(input_ids=inp, attention_mask=att).logits
            idx = torch.arange(len(batch), device=dev)
            ml = logits[idx, torch.tensor(posl, device=dev)].float()
            logp = torch.log_softmax(ml, dim=-1)
            tv, ti = logp.topk(args.topk, dim=-1)
            tv = tv.cpu().numpy(); ti = ti.cpu().numpy()
            for j, (di, g, ids, pos) in enumerate(batch):
                results[di][str(g)] = {'tok': ti[j].tolist(), 'logp': tv[j].tolist()}
            if bi % (args.batch * 10) == 0:
                print('batch', bi, '/', len(items), flush=True)

    cache = {}
    def decode(t):
        if t not in cache:
            cache[t] = norm_tok(tok.convert_ids_to_tokens(int(t)))
        return cache[t]
    with open(args.out, 'w', encoding='utf-8') as f:
        for di, r in enumerate(rows):
            pred = {}
            for g, d in results[di].items():
                ws, lps = [], []
                for t, lp in zip(d['tok'], d['logp']):
                    x = decode(t)
                    if not x:
                        continue
                    ws.append(x); lps.append(float(lp))
                if ws:
                    pred[g] = {'w': ws, 'lp': lps}
            f.write(json.dumps({'id': int(r['id']), 'pred': pred}) + '\n')
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
