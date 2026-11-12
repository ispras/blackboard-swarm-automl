import csv, random, sys, os, time

LENS=[12,8,5]
SAMPLE=302644
SEED=999

def toks_bytes(s):
    return tuple(s.encode('utf-8').split())

print("reading test...", flush=True)
test_ids=[]
with open('test_v2.txt', newline='', encoding='utf-8') as f:
    r=csv.reader(f); next(r)
    for row in r:
        test_ids.append((int(row[0]), row[1]))
print("test",len(test_ids),flush=True)
rng=random.Random(SEED)
sample=sorted(rng.sample(range(len(test_ids)), SAMPLE))
sample_set=set(sample)
ids=[i for i in sample_set]
test_toks={}
with open('test_v2.txt', newline='', encoding='utf-8') as f:
    r=csv.reader(f); next(r)
    for k,row in enumerate(r):
        if k in sample_set:
            test_toks[k]=toks_bytes(row[1])
print("sampled",len(test_toks),flush=True)

# maps anchor tuple -> list of test sample indices
pref={L:{} for L in LENS}
suff={L:{} for L in LENS}
for k,t in test_toks.items():
    for L in LENS:
        if len(t)>=L:
            pref[L].setdefault(t[:L],[]).append(k)
            suff[L].setdefault(t[-L:],[]).append(k)

CAP=60
counts={}  # (kind,L,anchor)->n stored
out=open('train_hits_full.tsv','w',encoding='utf-8',newline='')
w=csv.writer(out,delimiter='\t')
w.writerow(['test_idx','L','kind','line_idx','sentence'])

t0=time.time()
n_hits=0
with open('train_v2.txt','rb') as f:
    for ln,line in enumerate(f):
        if ln % 2000000 == 0:
            print("line",ln,"hits",n_hits,"t",round(time.time()-t0,1),flush=True)
        toks=tuple(line.rstrip(b'\r\n').split())
        m=len(toks)
        if m<7: continue
        for L in LENS:
            if m<L: continue
            a=toks[:L]
            if a in pref[L]:
                key=('p',L,a); c=counts.get(key,0)
                if c<CAP:
                    counts[key]=c+1
                    for k in pref[L][a]:
                        w.writerow([k,L,'p',ln,b' '.join(toks).decode('utf-8','replace')])
                        n_hits+=1
            b=toks[-L:]
            if b in suff[L]:
                key=('s',L,b); c=counts.get(key,0)
                if c<CAP:
                    counts[key]=c+1
                    for k in suff[L][b]:
                        w.writerow([k,L,'s',ln,b' '.join(toks).decode('utf-8','replace')])
                        n_hits+=1
        if False:
            break
out.close()
print("done hits",n_hits,"time",round(time.time()-t0,1),flush=True)
