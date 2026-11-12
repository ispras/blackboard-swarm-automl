import argparse, csv, json, os, time
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="roberta-large")
    ap.add_argument("--test",default="/download_job_code/payload/sandbox/workspaces/castor_iota/test_v2.txt")
    ap.add_argument("--outdir",default="mlm_test_parts")
    ap.add_argument("--topk",type=int,default=8)
    ap.add_argument("--batch",type=int,default=192)
    ap.add_argument("--maxlen",type=int,default=160)
    ap.add_argument("--chunksent",type=int,default=4000)
    ap.add_argument("--start",type=int,default=0)
    ap.add_argument("--end",type=int,default=10**9)
    args=ap.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(args.model,use_fast=True)
    model=AutoModelForMaskedLM.from_pretrained(args.model,torch_dtype=torch.float16).cuda().eval()
    mask_id=tok.mask_token_id
    print("loaded",args.model,"mask",mask_id,round(time.time()-t0,1),flush=True)

    # read all test sentences (csv)
    sents=[]
    with open(args.test,newline='',encoding='utf-8') as f:
        r=csv.reader(f); next(r)
        for row in r: sents.append(row[1])
    N=len(sents)
    print("sentences",N,flush=True)
    end=min(args.end,N)
    B=args.batch; tk=args.topk

    for cs in range(args.start,end,args.chunksent):
        ce=min(cs+args.chunksent,end)
        part=os.path.join(args.outdir,f"part_{cs:07d}_{ce:07d}.jsonl")
        if os.path.exists(part):
            print("skip existing",part,flush=True); continue
        tmp=part+".tmp"
        # build jobs for this chunk (no truncation before tokenization; tokenizer truncates)
        jobs=[]
        for sid in range(cs,ce):
            toks=sents[sid].split(); n=len(toks)
            if n<3: continue
            for j in range(1,n):
                text=" ".join(toks[:j])+" "+tok.mask_token+" "+" ".join(toks[j:])
                jobs.append((sid,j,text))
        enc=[]
        for sid,j,text in jobs:
            ids=tok(text,add_special_tokens=True,truncation=True,max_length=args.maxlen)["input_ids"]
            try: mp=ids.index(mask_id)
            except ValueError: continue
            enc.append((len(ids),sid,j,ids,mp))
        enc.sort(key=lambda x:x[0])
        del jobs
        results={}
        with torch.no_grad():
            for s in range(0,len(enc),B):
                chunk=enc[s:s+B]
                maxlen=max(x[0] for x in chunk)
                input_ids=torch.full((len(chunk),maxlen),tok.pad_token_id,dtype=torch.long)
                attn=torch.zeros((len(chunk),maxlen),dtype=torch.long)
                mpos=[]
                for r,(L,sid,j,ids,mp) in enumerate(chunk):
                    input_ids[r,:L]=torch.tensor(ids); attn[r,:L]=1; mpos.append(mp)
                input_ids=input_ids.cuda(); attn=attn.cuda()
                mpos_t=torch.tensor(mpos,device="cuda")
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    logits=model(input_ids,attention_mask=attn).logits
                sel=logits[torch.arange(len(chunk),device="cuda"),mpos_t,:].float()
                probs=torch.softmax(sel,dim=-1)
                topv,topi=torch.topk(probs,tk,dim=-1)
                topv=topv.cpu().numpy(); topi=topi.cpu().numpy()
                for r,(L,sid,j,ids,mp) in enumerate(chunk):
                    cands=[]
                    for v,ii in zip(topv[r],topi[r]):
                        w=tok.decode([int(ii)]).strip()
                        if w: cands.append([float(v),w])
                    results.setdefault(sid,[]).append([j,cands])
        with open(tmp,"w",encoding="utf-8") as f:
            for sid in sorted(results):
                f.write(json.dumps({"id":sid,"gaps":results[sid]},ensure_ascii=False)+"\n")
        os.replace(tmp,part)
        print(f"chunk {cs}-{ce} enc={len(enc)} done {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
