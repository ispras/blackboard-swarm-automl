import csv, json, random, zipfile, io, sys

TRAIN_ZIP = '../../data/train_v2.txt.zip'
OUT = '.'
N_VAL = 20000
SEED = 1234

rng = random.Random(SEED)
from collections import deque
buf = deque(maxlen=N_VAL)
with zipfile.ZipFile(TRAIN_ZIP) as z:
    name = z.namelist()[0]
    with z.open(name) as f:
        for i, raw in enumerate(f):
            buf.append(raw.decode('utf-8', 'replace').rstrip('\n'))
print('collected', len(buf), 'last train lines', file=sys.stderr)

rows = []
for line in buf:
    line = line.strip()
    if not line:
        continue
    toks = line.split(' ')
    toks = [t for t in toks if t != '']
    if len(toks) < 4:
        continue
    rows.append(toks)
print('valid val sentences', len(rows), file=sys.stderr)

val_test = []
val_truth = []
for rid, toks in enumerate(rows):
    n = len(toks)
    # missing word is never the first and never the last (last is '.')
    idx = rng.randrange(1, n - 1)
    w = toks[idx]
    deleted = toks[:idx] + toks[idx + 1:]
    val_test.append((rid, ' '.join(deleted)))
    val_truth.append({'id': rid, 'gap': idx, 'word': w, 'orig': ' '.join(toks)})

with open(f'{OUT}/val_test.csv', 'w', newline='', encoding='utf-8') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(['id', 'sentence'])
    for rid, s in val_test:
        wr.writerow([rid, s])

with open(f'{OUT}/val_truth.jsonl', 'w', encoding='utf-8') as f:
    for d in val_truth:
        f.write(json.dumps(d) + '\n')

import statistics
lens = [len(d['word']) for d in val_truth]
print('mean word len', statistics.mean(lens), 'median', statistics.median(lens), file=sys.stderr)
print('done', file=sys.stderr)
