import csv, json, os, time
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

TEST="/download_job_code/payload/sandbox/workspaces/castor_iota/test_v2.txt"
OUT="/download_job_code/payload/sandbox/workspaces/castor_iota/mlm_test_parts/part_900000_900533.jsonl"
MODEL="roberta-large"

t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
model=AutoModelForMaskedLM.from_pretrained(MODEL,torch_dtype=torch.float16).cuda().eval()
mask_id=tok.mask_token_id
rows=[]
with open(TEST,newline='',encoding='utf-8') as f:
    r=csv.reader(f); next(r)
    for row in r:
        toks=row[1].split()
        if len(toks)==2:
            rows.append((int(row[0]),toks))
print("short rows",len(rows),flush=True)
texts=[toks[0]+" "+tok.mask_token+" "+toks[1] for _,toks in rows]
res={}
with torch.no_grad():
    for s in range(0,len(texts),64):
        ch=texts[s:s+64]
        enc=[tok(t,add_special_tokens=True,truncation=True,max_length=32)["input_ids"] for t in ch]
        ml=max(len(x) for x in enc)
        ids=torch.full((len(ch),ml),tok.pad_token_id,dtype=torch.long); attn=torch.zeros((len(ch),ml),dtype=torch.long)
        mp=[]
        for r,x in enumerate(enc):
            ids[r,:len(x)]=torch.tensor(x); attn[r,:len(x)]=1; mp.append(x.index(mask_id))
        ids=ids.cuda();attn=attn.cuda(); mp=torch.tensor(mp,device='cuda')
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            logits=model(ids,attention_mask=attn).logits
        sel=logits[torch.arange(len(ch),device='cuda'),mp,:].float()
        probs=torch.softmax(sel,dim=-1)
        topv,topi=torch.topk(probs,8,dim=-1)
        topv=topv.cpu().numpy(); topi=topi.cpu().numpy()
        for r,(tid,_) in enumerate(rows[s:s+64]):
            cands=[]
            for v,ii in zip(topv[r],topi[r]):
                w=tok.decode([int(ii)]).strip()
                if w: cands.append([float(v),w])
            res[tid]=[[1,cands]]
tmp=OUT+'.tmp'
with open(tmp,'w',encoding='utf-8') as f:
    for tid in sorted(res):
        f.write(json.dumps({"id":tid,"gaps":res[tid]},ensure_ascii=False)+"\n")
os.replace(tmp,OUT)
print("wrote",OUT,"in",round(time.time()-t0,1),flush=True)
