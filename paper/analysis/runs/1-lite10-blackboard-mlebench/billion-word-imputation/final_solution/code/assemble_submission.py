import argparse, csv, json, glob, sys

ap=argparse.ArgumentParser()
ap.add_argument("--test",default="/download_job_code/payload/sandbox/workspaces/castor_iota/test_v2.txt")
ap.add_argument("--retrieval",default="/download_job_code/payload/sandbox/workspaces/castor_iota/retrieval_recovered.tsv")
ap.add_argument("--rerank",nargs="*",default=[],help="rerank jsonl files (id,cands[[gap,word,mean,total]]); later files override earlier")
ap.add_argument("--margin",type=float,default=0.08)
ap.add_argument("--out",default="submission_assembled.csv")
args=ap.parse_args()

rec={}
try:
    with open(args.retrieval,encoding='utf-8') as f:
        r=csv.reader(f,delimiter='\t'); next(r)
        for tid,pos,word,nv in r:
            rec[int(tid)]=(int(pos),word)
except FileNotFoundError:
    pass
print("retrieval overrides",len(rec),flush=True)

pred={}  # id -> (gap,word,score) with margin applied later
raw={}
for path in args.rerank:
    n=0
    for l in open(path,encoding='utf-8'):
        l=l.strip()
        if not l: continue
        d=json.loads(l)
        cands=d["cands"]
        if not cands: continue
        cs=sorted(cands,key=lambda c:c[2],reverse=True)
        margin=cs[0][2]-cs[1][2] if len(cs)>1 else 1.0
        raw[d["id"]]=(cs[0][0],cs[0][1],margin,cs[0][2])
        n+=1
    print("loaded",path,n,flush=True)

rows=[]
with open(args.test,newline='',encoding='utf-8') as f:
    r=csv.reader(f); next(r)
    for row in r:
        rows.append((int(row[0]),row[1]))
print("test rows",len(rows),flush=True)

n_retr=0; n_neural=0; n_id=0
out=open(args.out,'w',encoding='utf-8',newline='')
out.write('id,"sentence"\n')
w=csv.writer(out,quoting=csv.QUOTE_NONNUMERIC,lineterminator='\n')
for tid,s in rows:
    toks=s.split()
    if tid in rec:
        pos,word=rec[tid]
        w.writerow([tid,' '.join(toks[:pos]+[word]+toks[pos:])]); n_retr+=1
    elif tid in raw and raw[tid][2]>=args.margin:
        gap,word,margin,sc=raw[tid]
        if 0<=gap<=len(toks):
            w.writerow([tid,' '.join(toks[:gap]+[word]+toks[gap:])]); n_neural+=1
        else:
            w.writerow([tid,s]); n_id+=1
    else:
        w.writerow([tid,s]); n_id+=1
out.close()
print(f"retrieval={n_retr} neural={n_neural} identity={n_id} total={len(rows)}")
