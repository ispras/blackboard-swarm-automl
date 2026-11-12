import argparse, json, glob, os, time, csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap=argparse.ArgumentParser()
ap.add_argument("--cand_glob",required=True)
ap.add_argument("--out_dir",required=True)
ap.add_argument("--model",default="ft_gpt2medium")
ap.add_argument("--attn",default="sdpa")
ap.add_argument("--batch",type=int,default=192)
ap.add_argument("--maxlen",type=int,default=200)
ap.add_argument("--part_offset",type=int,default=0)
ap.add_argument("--limit_parts",type=int,default=100000)
args=ap.parse_args()

os.makedirs(args.out_dir,exist_ok=True)
files=sorted(glob.glob(args.cand_glob))[args.part_offset:args.part_offset+args.limit_parts]
print("parts",len(files),flush=True)
rows=list(csv.reader(open("/download_job_code/payload/sandbox/workspaces/castor_iota/test_v2.txt",encoding="utf-8")))
sent={}
for row in rows[1:]:
    if len(row)>=2: sent[int(row[0])]=row[1]

t0=time.time()
tok=AutoTokenizer.from_pretrained(args.model); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,attn_implementation=args.attn).cuda().eval()
print("loaded",args.model,round(time.time()-t0,1),flush=True)

def score_texts(texts):
    enc=[]
    for i,text in enumerate(texts):
        ids=tok(text,add_special_tokens=False,truncation=True,max_length=args.maxlen)["input_ids"]
        enc.append((len(ids),i,ids))
    enc.sort(key=lambda x:x[0]); out=[0.0]*len(texts); B=args.batch
    with torch.no_grad():
        for st in range(0,len(enc),B):
            chunk=enc[st:st+B]; maxlen=max(x[0] for x in chunk)
            ids=torch.full((len(chunk),maxlen),tok.eos_token_id,dtype=torch.long)
            attn=torch.zeros((len(chunk),maxlen),dtype=torch.long)
            for r,(L,i,x) in enumerate(chunk):
                ids[r,:L]=torch.tensor(x);attn[r,:L]=1
            ids=ids.cuda();attn=attn.cuda()
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                logits=model(ids,attention_mask=attn).logits
            lp=torch.log_softmax(logits,dim=-1)
            for r,(L,i,x) in enumerate(chunk):
                s=0.0
                for pos in range(1,L): s+=float(lp[r,pos-1,x[pos]])
                out[i]=s/max(1,L-1)
    return out

for fi,fpath in enumerate(files):
    outp=os.path.join(args.out_dir,os.path.basename(fpath))
    if os.path.exists(outp): print("skip",outp,flush=True); continue
    items=[]
    for line in open(fpath,encoding="utf-8"):
        d=json.loads(line); sid=d["id"]
        if sid not in sent: continue
        toks=sent[sid].split()
        for c in d["cands"]:
            gap,word=c[0],c[1]
            text=" ".join(toks[:gap]+[word]+toks[gap:])
            items.append((sid,c,text))
    sc=score_texts([it[2] for it in items])
    byid={}
    for (sid,c,text),s in zip(items,sc):
        base=c[4]; members=c[5:]  # base, xl, qwen, smol
        allv=[base]+list(members)+[s]
        blend5=sum(allv)/len(allv)
        byid.setdefault(sid,[]).append([c[0],c[1],blend5,0.0,base]+list(members)+[s])
    with open(outp+".tmp","w",encoding="utf-8") as f:
        for sid in sorted(byid): f.write(json.dumps({"id":sid,"cands":byid[sid]})+"\n")
    os.replace(outp+".tmp",outp)
    print(f"part {fi+1}/{len(files)} {os.path.basename(fpath)} n={len(byid)} elapsed {time.time()-t0:.0f}s",flush=True)
print("done",flush=True)
