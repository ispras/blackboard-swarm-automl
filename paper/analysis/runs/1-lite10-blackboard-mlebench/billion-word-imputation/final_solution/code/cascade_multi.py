import argparse, json, glob, os, time, csv, io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap=argparse.ArgumentParser()
ap.add_argument("--base_glob",required=True)
ap.add_argument("--out_dir",required=True)
ap.add_argument("--models",default="gpt2-xl,Qwen/Qwen2.5-1.5B,HuggingFaceTB/SmolLM2-1.7B")
ap.add_argument("--attn_map",default="gpt2-xl:sdpa,Qwen/Qwen2.5-1.5B:sdpa,HuggingFaceTB/SmolLM2-1.7B:sdpa")
ap.add_argument("--topk_gaps",type=int,default=3)
ap.add_argument("--batch",type=int,default=96)
ap.add_argument("--maxlen",type=int,default=200)
ap.add_argument("--limit_parts",type=int,default=100000)
ap.add_argument("--part_offset",type=int,default=0)
ap.add_argument("--skip_existing",action="store_true")
args=ap.parse_args()

os.makedirs(args.out_dir,exist_ok=True)
files=sorted(glob.glob(args.base_glob))[args.part_offset:args.part_offset+args.limit_parts]
print("parts",len(files),flush=True)

# load test sentences (castor's test file = same ids)
rows=list(csv.reader(open("/download_job_code/payload/sandbox/workspaces/castor_iota/test_v2.txt",encoding="utf-8")))
sent={}
for row in rows[1:]:
    if len(row)>=2: sent[int(row[0])]=row[1]
print("test sentences",len(sent),flush=True)

models=args.models.split(",")
attn_map=dict(x.split(":",1) for x in args.attn_map.split(","))
loaded=[]
t0=time.time()
for m in models:
    tk=AutoTokenizer.from_pretrained(m)
    tk.pad_token=tk.eos_token
    md=AutoModelForCausalLM.from_pretrained(m,torch_dtype=torch.float16,attn_implementation=attn_map.get(m,"sdpa")).cuda().eval()
    loaded.append((m,tk,md))
    print("loaded",m,round(time.time()-t0,1),flush=True)

def score_texts(tk,md,texts):
    enc=[]
    for i,text in enumerate(texts):
        ids=tk(text,add_special_tokens=False,truncation=True,max_length=args.maxlen)["input_ids"]
        enc.append((len(ids),i,ids))
    enc.sort(key=lambda x:x[0])
    out=[0.0]*len(texts)
    B=args.batch
    with torch.no_grad():
        for st in range(0,len(enc),B):
            chunk=enc[st:st+B]
            maxlen=max(x[0] for x in chunk)
            ids=torch.full((len(chunk),maxlen),tk.eos_token_id,dtype=torch.long)
            attn=torch.zeros((len(chunk),maxlen),dtype=torch.long)
            for r,(L,i,x) in enumerate(chunk):
                ids[r,:L]=torch.tensor(x);attn[r,:L]=1
            ids=ids.cuda();attn=attn.cuda()
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                logits=md(ids,attention_mask=attn).logits
            lp=torch.log_softmax(logits,dim=-1)
            for r,(L,i,x) in enumerate(chunk):
                s=0.0
                for pos in range(1,L): s+=float(lp[r,pos-1,x[pos]])
                out[i]=s/max(1,L-1)
    return out

for fi,fpath in enumerate(files):
    outp=os.path.join(args.out_dir,os.path.basename(fpath))
    if args.skip_existing and os.path.exists(outp):
        continue
    items=[]
    for line in open(fpath,encoding="utf-8"):
        d=json.loads(line)
        sid=d["id"]
        if sid not in sent: continue
        toks=sent[sid].split()
        cands=sorted(d["cands"],key=lambda c:-c[2])[:args.topk_gaps]
        for gap,word,bl,tl in cands:
            text=" ".join(toks[:gap]+[word]+toks[gap:])
            items.append((sid,gap,word,bl,text))
    texts=[it[4] for it in items]
    allsc=[score_texts(tk,md,texts) for (m,tk,md) in loaded]
    byid={}
    for i,(sid,gap,word,bl,text) in enumerate(items):
        extra=[allsc[j][i] for j in range(len(loaded))]
        blend=(bl+sum(extra))/(1+len(extra))
        byid.setdefault(sid,[]).append([gap,word,blend,0.0]+[bl]+extra)
    with open(outp+".tmp","w",encoding="utf-8") as f:
        for sid in sorted(byid):
            f.write(json.dumps({"id":sid,"cands":byid[sid]})+"\n")
    os.replace(outp+".tmp",outp)
    print(f"part {fi+1}/{len(files)} {os.path.basename(fpath)} n={len(byid)} elapsed {time.time()-t0:.0f}s",flush=True)
print("done",flush=True)
