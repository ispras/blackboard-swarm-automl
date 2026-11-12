import json, csv, math, zipfile, io, os
import Levenshtein
B='/download_job_code/payload/sandbox/workspaces/betelgeuse_tau'
O='/download_job_code/payload/sandbox/workspaces/fomalhaut_zeta'
THR=float(os.environ.get('THR','0.35')); ALPHA=float(os.environ.get('ALPHA','0.3'))
MPB=float(os.environ.get('MPB','0.05')); DEPTH=int(os.environ.get('DEPTH','3'))
OUTFILE=os.environ.get('OUTFILE',f'{O}/submission_v2_final.csv')
lm={}
for l in open(f'{B}/test_lm_v4.jsonl'):
    d=json.loads(l); lm[d['id']]=d
tri={}
for l in open(f'{O}/trigram_test_chosen_v2.jsonl'):
    d=json.loads(l); tri[d['id']]=d['tri']
margin={}
for l in open(f'{O}/fused_gap_test_v2.jsonl'):
    d=json.loads(l); margin[d['id']]=float(d['margin'])
sents={}
z=zipfile.ZipFile('/download_job_code/payload/sandbox/data/test_v2.txt.zip'); f=io.TextIOWrapper(z.open('test_v2.txt'),encoding='utf-8')
r=csv.reader(f); next(r)
for row in r: sents[int(row[0])]=row[1]
def triprob(i):
    d={}
    for w,c in tri.get(i,[]):
        c=float(c)
        if w and c>0: d[w]=d.get(w,0)+c
    s=sum(d.values()) or 1.0
    return {w:v/s for w,v in d.items()}
def medoid(P):
    pool=set(P)|{',','the','a','of','to','and','in','.'}
    for w in P:
        for k in range(1,DEPTH+1):
            if len(w)>k: pool.add(w[:k])
    best=None
    for x in pool:
        e=sum(p*Levenshtein.distance(x,w) for w,p in P.items())+MPB*(len(x)+4.57)
        if best is None or e<best[0]: best=(e,x)
    return best[1]
out=open(OUTFILE,'w',newline=''); w=csv.writer(out); w.writerow(['id','sentence']); nins=0
for i in sorted(sents):
    T=sents[i].split(' '); mg=margin.get(i,0.0); o=lm.get(i)
    if mg>=THR and o is not None and len(o['base'])>=2 and len(o['large'])>=2:
        P={}
        for ww,lp in o['base'][:15]:
            if ww: P[ww]=P.get(ww,0)+0.3*math.exp(float(lp))
        for ww,lp in o['large'][:15]:
            if ww: P[ww]=P.get(ww,0)+0.7*math.exp(float(lp))
        for ww,p in triprob(i).items(): P[ww]=P.get(ww,0)+ALPHA*p
        g=int(o['gap']); x=medoid(P)
        pred=' '.join(T[:g]+([x] if x else [])+T[g:]); nins+=1
    else:
        pred=' '.join(T)
    w.writerow([i,pred])
out.close()
print('wrote',OUTFILE,'inserted',nins,'thr',THR,'alpha',ALPHA,'mpb',MPB,'depth',DEPTH)
