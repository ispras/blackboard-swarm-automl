import sys, os, time
sys.path.append('/usr/local/lib/python3.11/site-packages')
import numpy as np, pandas as pd
from tokenizers import Tokenizer

ROOT = '/download_job_code/payload/sandbox'
WORK = f'{ROOT}/workspaces/rigel_mu'
MAXLEN = 320

tok = Tokenizer.from_file(f'{WORK}/bpe1024.json')
BOS = tok.token_to_id('[BOS]')
EOS = tok.token_to_id('[EOS]')
PAD = tok.token_to_id('[PAD]')
print('special', PAD, BOS, EOS, 'vocab', tok.get_vocab_size())

df = pd.read_csv(f'{ROOT}/data/train_labels.csv')
t0 = time.time()
encs = tok.encode_batch(df.InChI.tolist(), add_special_tokens=False)
print('tokenized in', time.time()-t0, 's')
lens = np.array([len(e.ids) for e in encs], dtype=np.int32)
maxl = int(lens.max())
print('maxlen', maxl, 'p99', np.percentile(lens, 99))
assert maxl <= MAXLEN, maxl
arr = np.zeros((len(df), MAXLEN), dtype=np.int16)
for i, e in enumerate(encs):
    ids = e.ids
    arr[i, 0] = BOS
    arr[i, 1:1+len(ids)] = ids
    arr[i, 1+len(ids)] = EOS
# length including BOS+EOS
seqlens = lens + 2
np.save(f'{WORK}/labels_tokens.npy', arr)
np.save(f'{WORK}/labels_seqlens.npy', seqlens.astype(np.int32))
df[['image_id', 'InChI']].to_csv(f'{WORK}/labels_meta.csv', index=False)

# holdout split (same RNG as before)
rng = np.random.RandomState(42)
idx = rng.permutation(len(df))
val_idx = idx[:20000]
val_ids = set(df.image_id.values[val_idx])
mask = df.image_id.isin(val_ids).values
print('val size', mask.sum(), 'train size', (~mask).sum())
np.save(f'{WORK}/is_val.npy', mask)
print('saved. total time', time.time()-t0)
