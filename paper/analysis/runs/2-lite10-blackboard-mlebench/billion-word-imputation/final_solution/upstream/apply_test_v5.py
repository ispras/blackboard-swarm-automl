import json, csv, math, time, pickle, os, zipfile, io
import numpy as np

BASE='/download_job_code/payload/sandbox/workspaces/betelgeuse_tau'
OUT='/download_job_code/payload/sandbox/workspaces/fomalhaut_zeta'
PUNCT={',','.',';',':','!','?','"',"'",'(',')','--','-','``',"''"}
def norm(t): return t.replace('\u0120',' ').replace('\u010a',' ').replace('\u0109',' ').strip()
t0=time.time()

# test sentences
sents={}
z=zipfile.ZipFile('/download_job_code/payload/sandbox/data/test_v2.txt.zip')
f=io.TextIOWrapper(z.open('test_v2.txt'),encoding='utf-8')
r=csv.reader(f); next(r)
for row in r: sents[int(row[0])]=row[1]
print('test sents',len(sents),flush=True)
fom={}
for l in open(f'{OUT}/fomalhaut_test_gap_probs_top10.jsonl'):
    d=json.loads(l); fom[d['id']]={int(g):float(p) for g,p in d['gap_probs']}
causal={}
for l in open(f'{OUT}/causal_surp_test.jsonl'):
    d=json.loads(l); causal[d['id']]=d['surp']
verif={}
for l in open('../mirfak_chi/verif_gaps_test.jsonl'):
    d=json.loads(l); verif[d['id']]={int(g):(float(r),float(p)) for g,r,p in d['gap_probs']}
verif2={}
for l in open('../mirfak_chi/verif_ft_gaps_test.jsonl'):
    d=json.loads(l); verif2[d['id']]={int(g):(float(r),float(p)) for g,r,p in d['gap_probs']}
verif3={}
for l in open(f'{OUT}/verif_large_gaps_test.jsonl'):
    d=json.loads(l); verif3[d['id']]={int(g):(float(r),float(p)) for g,r,p in d['gap_probs']}
print('verif test loaded',len(verif),'verif2',len(verif2),'verif3',len(verif3),flush=True)
pkl=pickle.load(open(f'{OUT}/final_gbdt_v5.pkl','rb')); clf=pkl['model']; THR=float(os.environ.get('THR','0.35'))
print('loaded model thr',THR,'t=%.1f'%(time.time()-t0),flush=True)

def fom_top10(mass):
    items=sorted(mass.items(),key=lambda kv:-kv[1])[:10]
    topd=dict(items); rest=max(0.0,1.0-sum(topd.values())); return topd,rest

def build(T, words, mass, cs, vd, vd2, vd3):
    n=len(T); topd,rest=fom_top10(mass)
    gaps_all=[]
    for gs,d in words.items():
        g=int(gs)
        if 1<=g<n: gaps_all.append((g,d))
    if not gaps_all: return None
    missing=[g for g,_ in gaps_all if g not in topd]
    floor=math.log(max(rest,1e-9)/max(1,len(missing))) if missing else -30.0
    if cs:
        arr=cs[1:] if len(cs)>1 else cs
        mu=sum(arr)/len(arr); sd=(sum((x-mu)**2 for x in arr)/len(arr))**0.5 or 1.0
        med=sorted(arr)[len(arr)//2] if arr else 0.0
    rows=[];gaps=[];kept=[]
    for g,d in gaps_all:
        pairs=[(norm(w),float(lp)) for w,lp in d if norm(w)][:20]
        if len(pairs)<2: continue
        ws=[w for w,_ in pairs]; lps=[lp for _,lp in pairs]
        top1=lps[0];top2=lps[1];l20=lps[-1]
        nxt=T[g];prv=T[g-1]
        lp_nxt=min(lps) if nxt not in ws else lps[ws.index(nxt)]
        lp_prv=min(lps) if prv not in ws else lps[ws.index(prv)]
        fl=math.log(max(topd[g],1e-12)) if g in topd else floor
        if cs:
            sg=cs[g] if g<len(cs) else 0.0; zsg=(sg-mu)/sd; rat=sg/(med+1e-6)
            sgp=cs[g-1] if g-1>=0 else 0.0
            rank=sum(1 for x in arr if x<sg)/max(1,len(arr))
        else: sg=zsg=rat=sgp=rank=0.0
        if vd:
            vr_all=[x[0] for x in vd.values()]; vmin=min(vr_all)
            vraw=vd[g][0] if g in vd else vmin-2.0
            vp=vd[g][1] if g in vd else 0.0
            vmu=sum(vr_all)/len(vr_all); vsd=(sum((x-vmu)**2 for x in vr_all)/len(vr_all))**0.5 or 1.0
            vz=(vraw-vmu)/vsd
        else: vraw=vp=vz=0.0
        if vd2:
            vr2=[x[0] for x in vd2.values()]; vmin2=min(vr2)
            vraw2=vd2[g][0] if g in vd2 else vmin2-2.0
            vp2=vd2[g][1] if g in vd2 else 0.0
            vmu2=sum(vr2)/len(vr2); vsd2=(sum((x-vmu2)**2 for x in vr2)/len(vr2))**0.5 or 1.0
            vz2=(vraw2-vmu2)/vsd2
        else: vraw2=vp2=vz2=0.0
        if vd3:
            vr3=[x[0] for x in vd3.values()]; vmin3=min(vr3)
            vraw3=vd3[g][0] if g in vd3 else vmin3-2.0
            vp3=vd3[g][1] if g in vd3 else 0.0
            vmu3=sum(vr3)/len(vr3); vsd3=(sum((x-vmu3)**2 for x in vr3)/len(vr3))**0.5 or 1.0
            vz3=(vraw3-vmu3)/vsd3
        else: vraw3=vp3=vz3=0.0
        rows.append([fl,top1,top2,l20,top1-l20,top1-top2,top1-lp_nxt,top1-lp_prv,
                     g/n,(n-g)/n,len(T[g-1]),len(T[g]),
                     float(T[g-1] in PUNCT),float(T[g] in PUNCT),
                     float(g==1),float(g==n-1),1,sg,zsg,rat,sgp,rank,vraw,vz,vp,vraw2,vz2,vp2,vraw3,vz3,vp3])
        gaps.append(g); kept.append(pairs)
    return rows,gaps,kept

files=sorted([f'{BASE}/{x}' for x in os.listdir(BASE) if x.startswith('mlm_test_') and x.endswith('.jsonl')])
print('files',[os.path.basename(x) for x in files],flush=True)
done={}
buf_rows=[];buf_meta=[];buf_ids=[]
def flush():
    global buf_rows,buf_meta,buf_ids
    if not buf_rows: return
    A=np.array(buf_rows,dtype=np.float32)
    s=clf.predict_proba(A)[:,1]
    k=0
    for i,gaps,kept in buf_meta:
        sc=s[k:k+len(gaps)]; k+=len(gaps)
        order=np.argsort(-sc); pg=gaps[order[0]]
        mg=float(sc[order[0]]-sc[order[1]]) if len(order)>1 else 99.0
        lg=np.log(np.clip(sc,1e-9,1-1e-9)/np.clip(1-sc,1e-9,1-1e-9))
        e=np.exp(lg-lg.max()); post=e/e.sum()
        done[i]={'gap':int(pg),'margin':mg,'p1':float(post[order[0]]),'words':kept[order[0]],
                 'gap_probs':[[int(gaps[j]),float(post[j])] for j in order[:10]],
                 'gap_scores':[[int(gaps[j]),float(sc[j])] for j in order[:10]]}
    buf_rows=[];buf_meta=[];buf_ids=[]
for fn in files:
    cnt=0
    for line in open(fn):
        d=json.loads(line); i=d['id']
        if i in done: continue
        T=sents[i].split(' ')
        b=build(T, d['words'], fom.get(i,{}), causal.get(i,[]), verif.get(i,{}), verif2.get(i,{}), verif3.get(i,{}))
        if b is None: continue
        rows,gaps,kept=b
        buf_rows.extend(rows); buf_meta.append((i,gaps,kept)); buf_ids.append(i)
        cnt+=1
        if len(buf_ids)>=4000: flush()
    flush(); print('file',os.path.basename(fn),'ids',cnt,'done',len(done),'t=%.1f'%(time.time()-t0),flush=True)
# fallback for not done: use fom top gap, margin 0 -> no-op at decision
with open(f'{OUT}/test_fused_top1_v5.jsonl','w') as fo:
    for i in sorted(sents):
        if i in done:
            d=done[i]; d['id']=i; fo.write(json.dumps(d)+'\n')
        else:
            gp=fom.get(i,{})
            pg=max(gp,key=gp.get) if gp else 1
            fo.write(json.dumps({'id':i,'gap':int(pg),'margin':0.0,'p1':0.0,'words':[],'gap_probs':[],'gap_scores':[]})+'\n')
print('uncovered', len(sents)-len(done), 'of', len(sents), flush=True)
print('wrote test_fused_top1_v5.jsonl t=%.1f'%(time.time()-t0),flush=True)
