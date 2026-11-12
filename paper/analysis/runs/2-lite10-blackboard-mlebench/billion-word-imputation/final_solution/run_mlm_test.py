"""roberta-base all-gap MLM stats for TEST (resumable shards).

For each sentence: for every gap g=1..n-1 insert [MASK] between T[g-1],T[g],
record top-20 conditional logp, then emit compact per-gap stats + top-15 words.
Output shard: mlm_test_{start}_{end}.jsonl
 row = {"id", "n", "gaps":[[g,top1,marg_next,marg_prev,spread,margin12],...],
        "words": {str(g): [[word,logp],...]}}
"""
import argparse, csv, io, json, sys, zipfile, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def norm_tok(t):
    return t.replace('\u0120', ' ').replace('\u010a', ' ').replace('\u0109', ' ').strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='roberta-base')
    ap.add_argument('--src', default='../../data/test_v2.txt.zip')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=40000)
    ap.add_argument('--batch', type=int, default=96)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--maxlen', type=int, default=256)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or f'mlm_test_{a.start}_{a.end}.jsonl'

    tok = AutoTokenizer.from_pretrained(a.model, add_prefix_space=True)
    model = AutoModelForMaskedLM.from_pretrained(a.model, torch_dtype=torch.float16)
    model.eval().cuda()
    mask_id = tok.mask_token_id
    dev = 'cuda'

    z = zipfile.ZipFile(a.src)
    rd = csv.reader(io.TextIOWrapper(z.open('test_v2.txt'), encoding='utf-8'))
    next(rd)
    rows = []
    for r in rd:
        if int(r[0]) < a.start:
            continue
        if int(r[0]) >= a.end:
            break
        rows.append((int(r[0]), r[1]))
    print('rows', len(rows), flush=True)

    def decode(t):
        return norm_tok(tok.convert_ids_to_tokens(int(t)))

    fout = open(out, 'w', encoding='utf-8')
    CH = 4000  # sentences per processing block
    for c0 in range(0, len(rows), CH):
        block = rows[c0:c0+CH]
        items = []
        for di, (rid, sent) in enumerate(block):
            words = [w for w in sent.split(' ') if w != '']
            enc = tok(words, is_split_into_words=True, add_special_tokens=True,
                      truncation=True, max_length=a.maxlen)
            ids = enc['input_ids']; wids = enc.word_ids()
            first = {}
            for ti, w in enumerate(wids):
                if w is not None and w not in first:
                    first[w] = ti
            nw = len(words)
            for g in range(1, nw):
                if g in first:
                    items.append((di, g, nw, ids, first[g]))
        res = [dict() for _ in block]
        with torch.no_grad():
            for bi in range(0, len(items), a.batch):
                batch = items[bi:bi+a.batch]
                maxlen = max(len(x[3]) + 1 for x in batch)
                inp = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(batch), maxlen), dtype=torch.long)
                posl = []
                for j, (di, g, nw, ids, pos) in enumerate(batch):
                    seq = ids[:pos] + [mask_id] + ids[pos:]
                    L = len(seq)
                    inp[j, :L] = torch.tensor(seq, dtype=torch.long)
                    att[j, :L] = 1
                    posl.append(pos)
                inp = inp.to(dev); att = att.to(dev)
                logits = model(input_ids=inp, attention_mask=att).logits
                idx = torch.arange(len(batch), device=dev)
                ml = logits[idx, torch.tensor(posl, device=dev)].float()
                del logits
                lp = torch.log_softmax(ml, dim=-1)
                tv, ti = lp.topk(a.topk, dim=-1)
                tv = tv.cpu().numpy(); ti = ti.cpu().numpy()
                for j, (di, g, nw, ids, pos) in enumerate(batch):
                    res[di][g] = (ti[j].tolist(), tv[j].tolist())
        # emit
        for di, (rid, sent) in enumerate(block):
            words = [w for w in sent.split(' ') if w != '']
            gaps = []; wd = {}
            for g in range(1, len(words)):
                if g not in res[di]:
                    continue
                tis, lps = res[di][g]
                wl = [(decode(t), float(v)) for t, v in zip(tis, lps)]
                wl = [(w, v) for w, v in wl if w]
                if len(wl) < 2:
                    continue
                ws = [w for w, _ in wl]; ls = [v for _, v in wl]
                mx = max(ls); top1 = ls[0]
                nxt = words[g]; prv = words[g-1]
                lp_nxt = (ls[ws.index(nxt)] if nxt in ws else min(ls))
                lp_prv = (ls[ws.index(prv)] if prv in ws else min(ls))
                gaps.append([g, top1, mx-lp_nxt, mx-lp_prv, top1-ls[-1], top1-ls[1]])
                wd[str(g)] = [[w, v] for w, v in wl[:15]]
            fout.write(json.dumps({'id': rid, 'n': len(words), 'gaps': gaps, 'words': wd}) + '\n')
        fout.flush()
        print('block', c0, '/', len(rows), flush=True)
    fout.close()
    print('wrote', out, flush=True)

if __name__ == '__main__':
    main()
