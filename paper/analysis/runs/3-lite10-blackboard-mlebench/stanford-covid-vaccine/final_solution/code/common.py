import json, os, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
WS = os.path.dirname(os.path.abspath(__file__))

SEQ_LEN = 107
SCORED_LEN = 68
TARGETS = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
ALL_TARGETS = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
SCORE_IDX = [ALL_TARGETS.index(t) for t in TARGETS]

SEQ_VOCAB = {c: i for i, c in enumerate('AGUC')}
STRUCT_VOCAB = {c: i for i, c in enumerate('.()')}
LOOP_VOCAB = {c: i for i, c in enumerate('SMIBHEX')}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def encode(rows):
    n = len(rows)
    seq = np.zeros((n, SEQ_LEN), dtype=np.int64)
    struct = np.zeros((n, SEQ_LEN), dtype=np.int64)
    loop = np.zeros((n, SEQ_LEN), dtype=np.int64)
    for i, r in enumerate(rows):
        for j, c in enumerate(r['sequence']):
            seq[i, j] = SEQ_VOCAB.get(c, 0)
        for j, c in enumerate(r['structure']):
            struct[i, j] = STRUCT_VOCAB.get(c, 0)
        for j, c in enumerate(r['predicted_loop_type']):
            loop[i, j] = LOOP_VOCAB.get(c, 0)
    return seq, struct, loop


def load_train():
    rows = load_jsonl(os.path.join(DATA, 'train.json'))
    seq, struct, loop = encode(rows)
    y = np.zeros((len(rows), SEQ_LEN, 5), dtype=np.float32)
    mask = np.zeros((len(rows), SEQ_LEN), dtype=np.float32)
    for i, r in enumerate(rows):
        for k, t in enumerate(ALL_TARGETS):
            v = r[t]
            y[i, :len(v), k] = v
            mask[i, :len(v)] = 1.0
    sn = np.array([r['signal_to_noise'] for r in rows], dtype=np.float32)
    ids = [r['id'] for r in rows]
    return dict(seq=seq, struct=struct, loop=loop, y=y, mask=mask, sn=sn, ids=ids, rows=rows)


def load_test():
    rows = load_jsonl(os.path.join(DATA, 'test.json'))
    seq, struct, loop = encode(rows)
    ids = [r['id'] for r in rows]
    return dict(seq=seq, struct=struct, loop=loop, ids=ids, rows=rows)


def mcrmse(y_true, y_pred, mask, cols=SCORE_IDX):
    # y_true, y_pred: (n, L, C) ; returns mean over cols of RMSE over masked positions
    errs = []
    for c in cols:
        d = (y_true[:, :, c] - y_pred[:, :, c]) ** 2
        errs.append(np.sqrt((d * mask).sum() / mask.sum()))
    return float(np.mean(errs)), [float(e) for e in errs]


def write_submission(pred, ids, path):
    # pred: (n, SEQ_LEN, 5) in ALL_TARGETS order
    sub = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))
    id2i = {s: i for i, s in enumerate(ids)}
    m = sub['id_seqpos'].str.rsplit('_', n=1)
    sub = sub.copy()
    sub['_id'] = [x[0] for x in m]
    sub['_pos'] = [int(x[1]) for x in m]
    for k, t in enumerate(ALL_TARGETS):
        sub[t] = [pred[id2i[i], p, k] for i, p in zip(sub['_id'], sub['_pos'])]
    sub = sub[['id_seqpos'] + ALL_TARGETS]
    sub.to_csv(path, index=False)
    return sub


def structure_features(rows):
    """Per-position extra features from structure/loop/position."""
    n = len(rows)
    L = SEQ_LEN
    feats = np.zeros((n, L, 8), dtype=np.float32)
    for i, r in enumerate(rows):
        st = r['structure']
        # pair matching
        stack = []
        pair = np.full(L, -1, dtype=np.int64)
        for j, c in enumerate(st):
            if c == '(':
                stack.append(j)
            elif c == ')' and stack:
                k = stack.pop()
                pair[k] = j
                pair[j] = k
        paired = (pair >= 0)
        # distance to nearest unpaired base
        dl = np.zeros(L, dtype=np.float32)
        last = -1000
        for j in range(L):
            if not paired[j]:
                last = j
            dl[j] = min(j - last, 100) / 100.0
        dr = np.zeros(L, dtype=np.float32)
        nxt = 10000
        for j in range(L - 1, -1, -1):
            if not paired[j]:
                nxt = j
            dr[j] = min(nxt - j, 100) / 100.0
        for j in range(L):
            feats[i, j, 0] = j / (L - 1)
            feats[i, j, 1] = float(paired[j])
            feats[i, j, 2] = abs(pair[j] - j) / L if paired[j] else 0.0
            feats[i, j, 3] = dl[j]
            feats[i, j, 4] = dr[j]
            feats[i, j, 5] = float(r['predicted_loop_type'][j] == 'S')
            feats[i, j, 6] = float(j < SCORED_LEN)
            feats[i, j, 7] = 1.0
    return feats


def load_train_extra():
    d = load_train()
    d['extra'] = structure_features(d['rows'])
    return d


def load_test_extra():
    d = load_test()
    d['extra'] = structure_features(d['rows'])
    return d
