import zipfile, io, json, csv, time, os, math
from collections import Counter
PATH='/download_job_code/payload/sandbox/data/train_v2.txt.zip'
BASE='/download_job_code/payload/sandbox/workspaces/betelgeuse_tau'
OUT='/download_job_code/payload/sandbox/workspaces/fomalhaut_zeta'
COUNT_LINES=int(os.environ.get('COUNT_LINES','20000000'))
TOPK=int(os.environ.get('TOPK','25'))
t0=time.time()
# test sentences
test={}
z=zipfile.ZipFile('/download_job_code/payload/sandbox/data/test_v2.txt.zip'); f=io.TextIOWrapper(z.open('test_v2.txt'),encoding='utf-8')
r=csv.reader(f); next(r)
for row in r: test[int(row[0])]=row[1]
val={int(rr['id']):rr['sentence'] for rr in csv.DictReader(open(f'{BASE}/val_test.csv'))}
# chosen gaps
tg={}
for l in open(os.environ.get('GAPF_TEST',f'{OUT}/fused_gap_test.tsv')):
    p=l.rstrip('\n').split('\t'); tg[int(p[0])]=int(p[1])
vg={}
for l in open(os.environ.get('GAPF_VAL',f'{OUT}/val_gaps_v3.jsonl')):
    d=json.loads(l); vg[d['id']]=d['gap']
# build queries
vocab={}
def vid(t):
    x=vocab.get(t)
    if x is None: x=len(vocab); vocab[t]=x
    return x
qtest={}; qval={}
allq={}
for i,g in tg.items():
    T=test[i].split(' '); key=(vid(T[g-1]),vid(T[g])); qtest[i]=key; allq[key]=None
for i,g in vg.items():
    T=val[i].split(' '); key=(vid(T[g-1]),vid(T[g])); qval[i]=key; allq[key]=None
V=len(vocab)
qkeys=set(a*V+b for (a,b) in allq)
mid={k:Counter() for k in qkeys}
print('vocab',V,'queries',len(qkeys),flush=True)
vget=vocab.get
z2=zipfile.ZipFile(PATH); f2=io.TextIOWrapper(z2.open('train_v2.txt'),encoding='utf-8')
n=0
for line in f2:
    s=line.rstrip('\n')
    if not s: continue
    toks=s.split(' '); ids=[vget(t) for t in toks]; L=len(ids)
    for i in range(L-2):
        a=ids[i]
        if a is None: continue
        b=ids[i+2]
        if b is None: continue
        key=a*V+b
        d=mid.get(key)
        if d is not None: d[toks[i+1]]+=1
    n+=1
    if n%2000000==0: print('count',n,'t=%.1f'%(time.time()-t0),flush=True)
    if n>=COUNT_LINES: break
print('done count',n,flush=True)
def dump(gaps, qmap, sents, fn):
    with open(fn,'w') as fo:
        for i in sorted(gaps):
            key=qmap[i]; k=key[0]*V+key[1]; d=mid.get(k,{})
            items=d.most_common(TOPK)
            fo.write(json.dumps({'id':i,'gap':gaps[i],'tri':[[w,int(c)] for w,c in items]})+'\n')
dump(tg,qtest,test,os.environ.get('OUT_TEST',f'{OUT}/trigram_test_chosen.jsonl'))
dump(vg,qval,val,os.environ.get('OUT_VAL',f'{OUT}/trigram_val_chosen.jsonl'))
print('wrote trigram for chosen gaps t=%.1f'%(time.time()-t0),flush=True)
