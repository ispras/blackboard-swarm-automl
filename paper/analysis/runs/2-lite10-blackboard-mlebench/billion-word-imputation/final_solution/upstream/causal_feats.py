import json, math, time, os, csv, zipfile, io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

MODEL = os.environ.get("MODEL", "gpt2-medium")
WHAT = os.environ.get("WHAT", "val")   # val|test|both
BASE = '/download_job_code/payload/sandbox/workspaces/betelgeuse_tau'
OUT = '/download_job_code/payload/sandbox/workspaces/fomalhaut_zeta'
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL).cuda().eval()
print('loaded', MODEL, flush=True)

def get_sentences(what):
    if what == 'val':
        vt = {int(r['id']): r['sentence'] for r in csv.DictReader(open(f'{BASE}/val_test.csv'))}
        return sorted(vt.items())
    z = zipfile.ZipFile('/download_job_code/payload/sandbox/data/test_v2.txt.zip')
    f = io.TextIOWrapper(z.open('test_v2.txt'), encoding='utf-8')
    r = csv.reader(f); next(r)
    return [(int(row[0]), row[1]) for row in r]

whats = ['val', 'test'] if WHAT == 'both' else [WHAT]
for what in whats:
    sents = get_sentences(what)
    items = [(len(s.split(' ')), i, s) for i, s in sents]
    items.sort()
    res = {}
    B = 64
    with torch.no_grad():
        for st in range(0, len(items), B):
            chunk = items[st:st+B]
            seqs = [it[2].split(' ') for it in chunk]
            enc0 = tok(seqs, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True, max_length=256)
            wids_all = [enc0.word_ids(j) for j in range(len(chunk))]
            enc = {k: v.cuda() for k, v in enc0.items()}
            out = model(**enc)
            logits = out.logits
            lp = torch.log_softmax(logits, dim=-1)
            ids = enc['input_ids']
            for j, it in enumerate(chunk):
                _, i, s = it
                n = len(s.split(' '))
                wids = wids_all[j]
                # surprisal per bpe position p (p>=1) = -lp[j,p-1,ids[j,p]]
                L = int(enc['attention_mask'][j].sum())
                surp = [0.0]*L
                for p in range(1, L):
                    t = ids[j, p]
                    surp[p] = float(-lp[j, p-1, t])
                word_first = {}
                for p in range(L):
                    w = wids[p]
                    if w is None: continue
                    if w not in word_first: word_first[w] = p
                ws = [surp[word_first[w]] if w in word_first else 0.0 for w in range(n)]
                res[i] = ws
            if st % (B*20) == 0:
                print(what, 'done', st, '/', len(items), 't=%.1f'%(time.time()-t0), flush=True)
    fn = f'{OUT}/causal_surp_{what}.jsonl'
    with open(fn, 'w') as fo:
        for i, ws in res.items():
            fo.write(json.dumps({'id': i, 'surp': ws})+'\n')
    print('wrote', fn, flush=True)
