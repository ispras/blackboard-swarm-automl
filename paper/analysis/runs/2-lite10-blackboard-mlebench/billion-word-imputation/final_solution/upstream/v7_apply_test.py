import json, csv, math, time, pickle, os, sys
import numpy as np
from v7_pipeline import build_features, load_extra, load_mass, norm, PUNCT

BASE='/download_job_code/payload/sandbox/workspaces/betelgeuse_tau'
FOM='/download_job_code/payload/sandbox/workspaces/fomalhaut_zeta'
VER='/download_job_code/payload/sandbox/workspaces/mirfak_chi'
OUT=VER
t0=time.time()

# test sentences
sents={}
with open(f'{VER}/test_v2.txt',newline='',encoding='utf-8') as f:
    r=csv.reader(f); next(r)
    for row in r: sents[int(row[0])]=row[1]
print('test sents',len(sents),flush=True)
fom=load_mass(f'{FOM}/fomalhaut_test_gap_probs_top10.jsonl')
causal={}
for l in open(f'{FOM}/causal_surp_test.jsonl'):
    d=json.loads(l); causal[d['id']]=d['surp']
v1=load_extra(f'{VER}/verif_gaps_test.jsonl')
v2=load_extra(f'{VER}/verif_ft_gaps_test.jsonl')
v3=load_extra(f'{BASE}/test_verif_large.jsonl')
_p2=f'{BASE}/test_verif_large_part2.jsonl'
if os.path.exists(_p2):
    for k,v in load_extra(_p2).items(): v3.setdefault(k,v)
print('loaded verifiers',len(v1),len(v2),len(v3),'t=%.1f'%(time.time()-t0),flush=True)
clf=pickle.load(open(f'{VER}/v7_gbdt.pkl','rb'))['model']

files=sorted([f'{BASE}/{x}' for x in os.listdir(BASE) if x.startswith('mlm_test_') and x.endswith('.jsonl')])
print('files',[os.path.basename(x) for x in files],flush=True)
done={}; buf_rows=[]; buf_meta=[]
def flush():
    global buf_rows,buf_meta
    if not buf_rows: return
    A=np.array(buf_rows,dtype=np.float32)
    s=clf.predict_proba(A)[:,1]; k=0
    for i,gaps,kept in buf_meta:
        sc=s[k:k+len(gaps)]; k+=len(gaps)
        order=np.argsort(-sc); pg=gaps[order[0]]
        mg=float(sc[order[0]]-sc[order[1]]) if len(order)>1 else 99.0
        lg=np.log(np.clip(sc,1e-9,1-1e-9)/np.clip(1-sc,1e-9,1-1e-9))
        e=np.exp(lg-lg.max()); post=e/e.sum()
        done[i]={'gap':int(pg),'margin':mg,'p1':float(post[order[0]]),'words':kept[order[0]],
                 'gap_probs':[[int(gaps[j]),float(post[j])] for j in order[:10]],
                 'gap_scores':[[int(gaps[j]),float(sc[j])] for j in order[:10]]}
    buf_rows=[]; buf_meta=[]
for fn in files:
    cnt=0
    for line in open(fn):
        d=json.loads(line); i=d['id']
        if i in done: continue
        T=sents[i].split(' ')
        rows,gaps=build_features(T, d['words'], fom.get(i,{}), causal.get(i,[]), v1.get(i,{}), v2.get(i,{}), v3.get(i,{}))
        if not rows: continue
        kept=[]
        # rebuild kept pairs aligned to gaps (same order as build_features)
        for g in gaps:
            dd=d['words'].get(str(g))
            if isinstance(dd,dict):
                pairs=[(norm(w),lp) for w,lp in zip(dd['w'],dd['lp']) if norm(w)]
            else:
                pairs=[(norm(w),lp) for w,lp in (dd or []) if norm(w)]
            kept.append(pairs[:20])
        buf_rows.extend(rows); buf_meta.append((i,gaps,kept)); cnt+=1
        if len(buf_meta)>=4000: flush()
    flush(); print('file',os.path.basename(fn),'ids',cnt,'done',len(done),'t=%.1f'%(time.time()-t0),flush=True)
with open(f'{OUT}/fused_gap_test_v7.jsonl','w') as fo, open(f'{OUT}/fused_gap_test_v7.tsv','w') as ft:
    for i in sorted(sents):
        if i in done:
            d=done[i]; d['id']=i; fo.write(json.dumps(d)+'\n')
            ft.write(f"{i}\t{d['gap']}\t{d['margin']}\n")
        else:
            gp=fom.get(i,{}); pg=max(gp,key=gp.get) if gp else 1
            fo.write(json.dumps({'id':i,'gap':int(pg),'margin':0.0,'p1':0.0,'words':[],'gap_probs':[],'gap_scores':[]})+'\n')
            ft.write(f"{i}\t{int(pg)}\t0.0\n")
print('uncovered',len(sents)-len(done),'of',len(sents),flush=True)
print('wrote fused_gap_test_v7 t=%.1f'%(time.time()-t0),flush=True)
