import json, math, statistics, time
import Levenshtein
B='/download_job_code/payload/sandbox/workspaces/betelgeuse_tau'
O='/download_job_code/payload/sandbox/workspaces/fomalhaut_zeta'
truth={json.loads(l)['id']:json.loads(l) for l in open(f'{B}/val_truth.jsonl')}
lm={}
for l in open(f'{B}/val_v4_lm.jsonl'):
    d=json.loads(l); lm[d['id']]=d
tri={}
for l in open(f'{O}/trigram_val_chosen_v2.jsonl'):
    d=json.loads(l); tri[d['id']]=d['tri']
margin={}
for l in open(f'{O}/val_gaps_v4.jsonl'):
    d=json.loads(l); margin[d['id']]=d['margin']
t0=time.time()
def fw(i):
    d={}
    for it in tri.get(i,[]):
        w=it[0]; c=float(it[1])
        if w and c>0: d[w]=d.get(w,0)+c
    z=sum(d.values()) or 1.0
    return {w:v/z for w,v in d.items()}
cache={}
for i,d in lm.items():
    base=[(w,float(lp)) for w,lp in d['base'][:12] if w]
    large=[(w,float(lp)) for w,lp in d['large'][:12] if w]
    cache[i]=(d['gap'],base,large,fw(i))
def medoid(P,mpb,cands_pool):
    best=None
    for x in cands_pool:
        e=0.0
        for w,p in P.items(): e+=p*Levenshtein.distance(x,w)
        e+=mpb*(len(x)+4.57)
        if best is None or e<best[0]: best=(e,x)
    return best[1]
def build_pool(P,depth):
    pool=set(P.keys())|{',','the','a','of','to','and','in','.'}
    for w in P:
        for k in range(1,depth+1):
            if len(w)>k: pool.add(w[:k])
    return list(pool)
def run(ids,alpha,mpb,depth,thr):
    ds=[]
    for i in ids:
        g,base,large,ft=cache[i]
        T=[t for t in truth[i]['orig'].split(' ') if t]
        tgi=truth[i]['gap']
        ds0=T[:tgi]+T[tgi+1:]
        noop=' '.join(ds0)
        if margin[i]<thr:
            ds.append(Levenshtein.distance(truth[i]['orig'],noop)); continue
        P={}
        for w,lp in base: P[w]=P.get(w,0)+0.3*math.exp(lp)
        for w,lp in large: P[w]=P.get(w,0)+0.7*math.exp(lp)
        if alpha>0:
            for w,p in ft.items(): P[w]=P.get(w,0)+alpha*p
        pool=build_pool(P,depth)
        x=medoid(P,mpb,pool)
        pred=' '.join(ds0[:g]+([x] if x else [])+ds0[g:])
        ds.append(Levenshtein.distance(truth[i]['orig'],pred))
    return statistics.mean(ds)
ids=sorted(lm)
tr=[i for i in ids if i<5000]; ev=[i for i in ids if i>=5000]
for alpha in [0.0,0.3,0.4,0.5]:
    print('train alpha %.2f depth3 mpb.05 th.35 -> %.4f'%(alpha,run(tr,alpha,0.05,3,0.35)),flush=True)
for alpha in [0.0,0.3,0.4,0.5]:
    for th in [0.30,0.35,0.40]:
        print('HELDOUT alpha %.2f depth3 mpb.05 th %.2f -> %.4f'%(alpha,th,run(ev,alpha,0.05,3,th)),flush=True)
