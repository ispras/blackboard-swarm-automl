import random, json, hashlib, sys

TRAIN = '/download_job_code/payload/sandbox/workspaces/antares_upsilon/train_v2.txt'
OUT = '/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout'
N_HOLD = 30000
SEED = 20241031

def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    # find total lines quickly by reading tail chunks: use known count but verify
    # We use the last N_HOLD lines via deque
    from collections import deque
    dq = deque(maxlen=N_HOLD)
    with open(TRAIN, 'r', encoding='utf-8') as f:
        for line in f:
            dq.append(line.rstrip('\n'))
    orig = list(dq)
    assert len(orig) == N_HOLD
    rng = random.Random(SEED)
    with open(f'{OUT}/holdout_orig.txt','w',encoding='utf-8') as fo, \
         open(f'{OUT}/holdout_gap.txt','w',encoding='utf-8') as fg, \
         open(f'{OUT}/holdout_meta.tsv','w',encoding='utf-8') as fm:
        fm.write('line\tn\trm_idx\trm_word\n')
        nvalid=0
        for i,s in enumerate(orig):
            toks = s.split(' ')
            n = len(toks)
            if n < 3:
                fo.write(s+'\n'); fg.write(s+'\n'); fm.write(f'{i}\t{n}\t-1\t\n'); continue
            j = rng.randint(1, n-2)  # interior index, 0-based, not first/last
            w = toks[j]
            gap = ' '.join(toks[:j] + toks[j+1:])
            fo.write(s+'\n'); fg.write(gap+'\n'); fm.write(f'{i}\t{n}\t{j}\t{w}\n')
            nvalid+=1
    print('wrote', N_HOLD, 'lines; valid', nvalid)

if __name__=='__main__':
    main()
