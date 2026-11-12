import zipfile, io, math, time, json, os, csv
PATH = '/download_job_code/payload/sandbox/data/train_v2.txt.zip'
TEST = '/download_job_code/payload/sandbox/data/test_v2.txt.zip'
BASE = '/download_job_code/payload/sandbox/workspaces/betelgeuse_tau'
OUT = '/download_job_code/payload/sandbox/workspaces/fomalhaut_zeta'
COUNT_LINES = int(os.environ.get("COUNT_LINES", "20000000"))
TOPG = int(os.environ.get("TOPG", "10"))
t0 = time.time()

truth = [json.loads(l) for l in open(f'{BASE}/val_truth.jsonl', encoding='utf-8')]
valtest = {int(r['id']): r['sentence'] for r in csv.DictReader(open(f'{BASE}/val_test.csv', encoding='utf-8'))}
val_ids = [d['id'] for d in truth]
val_toks = [valtest[i].split(' ') for i in val_ids]
true_gap = {d['id']: d['gap'] for d in truth}

zt = zipfile.ZipFile(TEST); ft = io.TextIOWrapper(zt.open('test_v2.txt'), encoding='utf-8')
r = csv.reader(ft); next(r)
test_ids = []; test_toks = []
for row in r:
    test_ids.append(int(row[0])); test_toks.append(row[1].split(' '))
print('val', len(val_toks), 'test', len(test_toks), flush=True)

vocab = {}
all_toks = val_toks + test_toks
for toks in all_toks:
    for t in toks:
        if t not in vocab: vocab[t] = len(vocab)
V = len(vocab)
ids_val = [[vocab[t] for t in toks] for toks in val_toks]
ids_test = [[vocab[t] for t in toks] for toks in test_toks]
print('vocab', V, 't=%.1f'%(time.time()-t0), flush=True)

qc = {}
for ids in ids_val + ids_test:
    for k in range(len(ids)-1):
        key = ids[k]*V + ids[k+1]
        qc[key] = 0
qmed = {k: 0 for k in qc}
print('query pairs', len(qc), flush=True)

z = zipfile.ZipFile(PATH); f = io.TextIOWrapper(z.open('train_v2.txt'), encoding='utf-8')
vget = vocab.get
n = 0
for line in f:
    s = line.rstrip('\n')
    if not s: continue
    ids = [vget(t) for t in s.split(' ')]
    L = len(ids)
    for i in range(L-1):
        a = ids[i]
        if a is None: continue
        b = ids[i+1]
        if b is None: continue
        key = a*V + b
        c = qc.get(key)
        if c is not None: qc[key] = c+1
    for i in range(L-2):
        a = ids[i]
        if a is None: continue
        b = ids[i+2]
        if b is None: continue
        key = a*V + b
        c = qmed.get(key)
        if c is not None: qmed[key] = c+1
    n += 1
    if n % 2000000 == 0: print('pass', n, 't=%.1f'%(time.time()-t0), flush=True)
    if n >= COUNT_LINES: break
print('count done', n, 't=%.1f'%(time.time()-t0), flush=True)

def scores(ids):
    nT = len(ids)
    gaplist = list(range(1, nT))
    sc = []
    for g in gaplist:
        key = ids[g-1]*V + ids[g]
        sc.append(math.log((qmed.get(key,0)+0.5)/(qc.get(key,0)+0.5)))
    m = max(sc) if sc else 0.0
    e = [math.exp(x-m) for x in sc]; ssum = sum(e) or 1.0
    pr = [x/ssum for x in e]
    order = sorted(range(len(sc)), key=lambda i: -sc[i])
    return gaplist, pr, order

# val: all gaps
top1 = top3 = top5 = 0
with open(f'{OUT}/fomalhaut_val_gap_probs20m.jsonl','w') as fo:
    for sid, ids in zip(val_ids, ids_val):
        gaplist, pr, order = scores(ids)
        fo.write(json.dumps({'id': sid, 'gap_probs': [[gaplist[i], pr[i]] for i in range(len(gaplist))]})+'\n')
        tg = true_gap[sid]
        rank = order.index(tg-1) if (tg-1) in order else 999
        top1 += rank==0; top3 += rank<3; top5 += rank<5
N=len(val_ids)
print('VAL top1 %.4f top3 %.4f top5 %.4f'%(top1/N, top3/N, top5/N), flush=True)

# test: top-k
with open(f'{OUT}/fomalhaut_test_gap_probs_top10.jsonl','w') as fo:
    for sid, ids in zip(test_ids, ids_test):
        gaplist, pr, order = scores(ids)
        top = [[gaplist[i], pr[i]] for i in order[:TOPG]]
        fo.write(json.dumps({'id': sid, 'gap_probs': top})+'\n')
print('done t=%.1f'%(time.time()-t0), flush=True)
