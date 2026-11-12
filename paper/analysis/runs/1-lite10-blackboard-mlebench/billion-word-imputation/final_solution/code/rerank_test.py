import argparse, csv, json, os, time, glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap=argparse.ArgumentParser()
ap.add_argument("--model",default="gpt2-xl")
ap.add_argument("--test",default="/download_job_code/payload/sandbox/workspaces/castor_iota/test_v2.txt")
ap.add_argument("--mlm-glob",default="/download_job_code/payload/sandbox/workspaces/castor_iota/mlm_test_parts/part_*.jsonl")
ap.add_argument("--outdir",default="rerank_test_parts")
ap.add_argument("--topw",type=int,default=1)
ap.add_argument("--batch",type=int,default=128)
ap.add_argument("--maxlen",type=int,default=200)
ap.add_argument("--topk-save",type=int,default=5)
ap.add_argument("--limit-parts",type=int,default=0)
args=ap.parse_args()

os.makedirs(args.outdir,exist_ok=True)
t0=time.time()
tok=AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token=tok.eos_token if tok.eos_token is not None else tok.pad_token
model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16).cuda().eval()
print("loaded",args.model,round(time.time()-t0,1),flush=True)

sents=[]
with open(args.test,newline='',encoding='utf-8') as f:
    r=csv.reader(f); next(r)
    for row in r: sents.append(row[1])

files=sorted(glob.glob(args.mlm_glob))
if args.limit_parts: files=files[:args.limit_parts]
print("mlm parts",len(files),flush=True)

def score(texts):
    enc=[]
    for i,text in enumerate(texts):
        ids=tok(text,add_special_tokens=False,truncation=True,max_length=args.maxlen)["input_ids"]
        if len(ids)>=2: enc.append((len(ids),i,ids))
    enc.sort(key=lambda x:x[0])
    out=[None]*len(texts)
    B=args.batch
    with torch.no_grad():
        for st in range(0,len(enc),B):
            chunk=enc[st:st+B]
            maxlen=max(x[0] for x in chunk)
            ids=torch.full((len(chunk),maxlen),tok.pad_token_id,dtype=torch.long)
            attn=torch.zeros((len(chunk),maxlen),dtype=torch.long)
            for r,(L,i,x) in enumerate(chunk):
                ids[r,:L]=torch.tensor(x); attn[r,:L]=1
            ids=ids.cuda();attn=attn.cuda()
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                logits=model(ids,attention_mask=attn).logits.float()
            logp=torch.log_softmax(logits,dim=-1)
            lp=torch.gather(logp[:,:-1,:],2,ids[:,1:].unsqueeze(-1)).squeeze(-1)
            mask=attn[:,1:].float()
            total=(lp*mask).sum(1); cnt=mask.sum(1).clamp(min=1); mean=total/cnt
            total=total.cpu().tolist(); mean=mean.cpu().tolist()
            for r,(L,i,x) in enumerate(chunk):
                out[i]=(mean[r],total[r])
    return out

for fi,fpath in enumerate(files):
    base=os.path.basename(fpath).replace('.jsonl','')
    outp=os.path.join(args.outdir,base+'.jsonl')
    if os.path.exists(outp):
        print("skip",outp,flush=True); continue
    dd=[json.loads(l) for l in open(fpath,encoding='utf-8')]
    cands=[]  # (sid,gap,word,text)
    for d in dd:
        sid=d["id"]; toks=sents[sid].split()
        for gap,cw in d["gaps"]:
            for _,w in cw[:args.topw]:
                if w: cands.append((sid,gap,w," ".join(toks[:gap]+[w]+toks[gap:])))
    texts=[c[3] for c in cands]
    tt=time.time()
    sc=score(texts)
    res={}
    for (sid,gap,w,_),s in zip(cands,sc):
        if s is None: continue
        res.setdefault(sid,[]).append([gap,w,s[0],s[1]])
    tmp=outp+".tmp"
    with open(tmp,'w',encoding='utf-8') as f:
        for sid in sorted(res):
            cs=sorted(res[sid],key=lambda c:c[2],reverse=True)
            margin=(cs[0][2]-cs[1][2]) if len(cs)>1 else 1.0
            f.write(json.dumps({"id":sid,"cands":cs,"margin":margin,"best":cs[0]})+"\n")
    os.replace(tmp,outp)
    print(f"part {fi+1}/{len(files)} {base} cands={len(cands)} {time.time()-tt:.0f}s",flush=True)
print("ALL DONE",round(time.time()-t0,1),flush=True)
