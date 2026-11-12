import gzip, pickle, sys, time
inp=sys.argv[1]; out=sys.argv[2]
t0=time.time()
# stream, keep per-id dict of best-gain entry per gap
recs={}  # id -> {gap: (word,gain,score)}
last=None; cur=None
with gzip.open(inp,'rt',encoding='utf-8') as f:
    for line in f:
        sid,k,w,g,sc=line.rstrip('\n').split('\t')
        k=int(k); g=float(g); sc=float(sc)
        if sid!=last:
            if last is not None: recs[int(last)]=cur
            last=sid; cur={}
        if k not in cur or g>cur[k][1]:
            cur[k]=(w,g,sc)
if last is not None: recs[int(last)]=cur
N=max(recs)+1
lst=[recs.get(i,{}) for i in range(N)]
pickle.dump(lst,open(out,'wb'),protocol=4)
print('wrote',out,'N',N,'total gaps',sum(len(d) for d in lst),'time',round(time.time()-t0,1))
