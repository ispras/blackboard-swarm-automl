import os, sys, time, random, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from functools import partial
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, get_linear_schedule_with_warmup

TRAIN="/download_job_code/payload/sandbox/workspaces/arcturus_kappa/train_v2.txt"
HOLDOUT_START=29998384-30000

class DS(IterableDataset):
    def __init__(self, tok, max_lines, w2i, seed=17, max_words=100, shuffle_buf=100000):
        self.tok=tok; self.max_lines=max_lines; self.w2i=w2i; self.seed=seed
        self.max_words=max_words; self.shuffle_buf=shuffle_buf
    def __iter__(self):
        rnd=random.Random(self.seed+torch.initial_seed()%(2**31))
        buf=[]
        with open(TRAIN,encoding='utf-8',errors='replace') as f:
            for i,line in enumerate(f):
                if i>=self.max_lines: break
                words=line.rstrip('\n').split()
                n=len(words)
                if n<3 or n>self.max_words: continue
                j=rnd.randint(1,n-2)
                target=words[j]
                inp=words[:j]+words[j+1:]
                text=" ".join([w if k==0 else "<mask> "+w for k,w in enumerate(inp)])
                wid=self.w2i.get(target,-1)
                buf.append((text,j-1,wid))
                if len(buf)>=self.shuffle_buf:
                    rnd.shuffle(buf)
                    for x in buf: yield x
                    buf=[]
        rnd.shuffle(buf)
        for x in buf: yield x

def collate(items, tok):
    texts=[x[0] for x in items]
    enc=tok(texts,padding=True,truncation=True,max_length=128,return_tensors='pt')
    return enc['input_ids'],enc['attention_mask'],torch.tensor([x[1] for x in items]),torch.tensor([x[2] for x in items])

class Detector(nn.Module):
    def __init__(self, base, Vw):
        super().__init__()
        self.encoder=base
        H=base.config.hidden_size
        self.gap_head=nn.Linear(H,1)
        self.word_head=nn.Linear(H,Vw)
    def forward(self,input_ids,attn):
        h=self.encoder(input_ids=input_ids,attention_mask=attn).last_hidden_state
        return h

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--init',default='roberta-base')
    ap.add_argument('--max_lines',type=int,default=5000000)
    ap.add_argument('--bs',type=int,default=192)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--epochs',type=int,default=1)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--out',default='detector_ft')
    ap.add_argument('--log',default='det_train.log')
    ap.add_argument('--steps_limit',type=int,default=0)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--word_lambda',type=float,default=1.0)
    ap.add_argument('--gpu_frac',type=float,default=0.0)
    args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    dev=args.device
    if dev=='cuda' and args.gpu_frac>0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac,0)
    words=json.load(open('wordvocab.json')); w2i={w:i for i,w in enumerate(words)}; Vw=len(words)
    tok=AutoTokenizer.from_pretrained(args.init)
    if args.init=='roberta-base':
        enc=AutoModel.from_pretrained('roberta-base')
    else:
        enc=AutoModelForMaskedLM.from_pretrained(args.init).roberta
    model=Detector(enc,Vw).to(dev); model.train()
    mask_id=tok.mask_token_id
    ds=DS(tok,args.max_lines,w2i); dl=DataLoader(ds,batch_size=args.bs,num_workers=args.workers,pin_memory=True,prefetch_factor=(4 if args.workers>0 else None),drop_last=True,collate_fn=partial(collate,tok=tok))
    steps_per_epoch=max(1,args.max_lines//args.bs); total=steps_per_epoch*args.epochs
    if args.steps_limit>0: total=min(total,args.steps_limit)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    sched=get_linear_schedule_with_warmup(opt,int(0.03*total),total)
    scaler=torch.cuda.amp.GradScaler(enabled=(dev=='cuda'))
    logf=open(args.log,'a')
    def log(m):
        print(m,flush=True); logf.write(m+'\n'); logf.flush()
    log(f'START init={args.init} max_lines={args.max_lines} bs={args.bs} lr={args.lr} total={total}')
    t0=time.time(); step=0; rl=0.0; rg=0.0; rw=0.0; rn=0
    for input_ids,attn,glab,wlab in dl:
        input_ids=input_ids.to(dev,non_blocking=True); attn=attn.to(dev,non_blocking=True)
        glab=glab.to(dev); wlab=wlab.to(dev)
        with torch.autocast(device_type='cuda' if dev=='cuda' else 'cpu', dtype=torch.float16, enabled=(dev=='cuda')):
            h=model(input_ids,attn)  # [B,T,H]
        B=h.size(0)
        gap_logits=[]; word_h=[]
        for b in range(B):
            pos=(input_ids[b]==mask_id).nonzero(as_tuple=True)[0]
            if len(pos)==0:
                gap_logits.append(torch.zeros(1,device=dev)); word_h.append(h[b,0]); continue
            gap_logits.append(model.gap_head(h[b,pos]).squeeze(-1))
            gi=int(glab[b]); gi=min(gi,len(pos)-1)
            word_h.append(h[b,pos[gi]])
        maxm=max(x.size(0) for x in gap_logits)
        gl=torch.full((B,maxm),-1e4,device=dev)
        for b,x in enumerate(gap_logits): gl[b,:x.size(0)]=x
        wh=torch.stack(word_h,0)
        lg=F.cross_entropy(gl,glab.clamp(max=maxm-1))
        wl_mask=(wlab>=0)
        if wl_mask.any():
            wl=F.cross_entropy(model.word_head(wh[wl_mask]).float(),wlab[wl_mask])
        else:
            wl=torch.zeros((),device=dev)
        loss=lg+args.word_lambda*wl
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step+=1; rl+=loss.item(); rg+=lg.item(); rw+=float(wl); rn+=1
        if step%50==0:
            el=time.time()-t0
            log(f'step {step}/{total} loss {rl/rn:.4f} gaploss {rg/rn:.4f} wordloss {rw/rn:.4f} {step/el:.2f} it/s {el/60:.1f}m')
            rl=rg=rw=0.0; rn=0
        if step%1000==0 or step==total:
            os.makedirs(args.out,exist_ok=True)
            ck=os.path.join(args.out,f'ckpt_{step}')
            torch.save({'encoder':model.encoder.state_dict(),'gap_head':model.gap_head.state_dict(),'word_head':model.word_head.state_dict(),'words':words},ck+'.pt')
            log('SAVED '+ck)
        if step>=total: break
    log(f'DONE step={step}')

if __name__=='__main__': main()
