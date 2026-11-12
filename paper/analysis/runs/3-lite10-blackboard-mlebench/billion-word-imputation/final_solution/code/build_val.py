import random, json
random.seed(12345)
# validation = last 12000 lines of train (disjoint from finetune slice which will use first ~10M lines)
path='train_v2.txt'
# count total quickly? we know 29998384
import os
N=29998384
start=N-12000
refs=[]; dels=[]; meta=[]
with open(path) as f:
    for i,line in enumerate(f):
        if i<start: continue
        line=line.rstrip('\n')
        t=line.split()
        if len(t)<=2: continue
        j=random.randint(1,len(t)-2)
        w=t[j]
        dt=t[:j]+t[j+1:]
        refs.append(line)
        dels.append(' '.join(dt))
        meta.append({'j':j,'w':w,'L':len(t)})
with open('val_ref.txt','w') as f: f.write('\n'.join(refs)+'\n')
with open('val_del.txt','w') as f: f.write('\n'.join(dels)+'\n')
with open('val_meta.jsonl','w') as f:
    for m in meta: f.write(json.dumps(m)+'\n')
print('val size',len(refs))
