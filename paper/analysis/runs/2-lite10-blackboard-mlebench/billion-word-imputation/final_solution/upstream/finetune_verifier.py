import argparse, numpy as np, torch, torch.nn as nn, time
from transformers import AutoTokenizer, AutoModel

class Model(nn.Module):
    def __init__(self,name):
        super().__init__()
        self.enc=AutoModel.from_pretrained(name)
        H=self.enc.config.hidden_size
        self.head=nn.Sequential(nn.Linear(H,512),nn.GELU(),nn.Dropout(0.15),nn.Linear(512,1))
    def forward(self,ids,att,poss):
        h=self.enc(input_ids=ids,attention_mask=att).last_hidden_state
        idx=torch.arange(len(ids),device=ids.device)
        hv=h[idx,torch.tensor(poss,device=ids.device)]
        return self.head(hv).squeeze(-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='corrupt_train.npz')
    ap.add_argument('--model',default='roberta-base')
    ap.add_argument('--out',default='verif_ft.pt')
    ap.add_argument('--epochs',type=int,default=2)
    ap.add_argument('--nsent',type=int,default=480000)
    ap.add_argument('--nval',type=int,default=2000)
    ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--nneg',type=int,default=2)
    ap.add_argument('--lr',type=float,default=2e-5)
    ap.add_argument('--maxtok',type=int,default=96)
    args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=True
    d=np.load(args.data)
    ids=d['ids']; ws=d['wstart']; nw=d['nw']; tg=d['tg']; ln=d['ln']
    tok=AutoTokenizer.from_pretrained(args.model, **({"add_prefix_space":True} if ("roberta" in args.model or "gpt2" in args.model) else {}))
    mask_id=tok.mask_token_id; pad_id=tok.pad_token_id or 1
    N=min(args.nsent,len(ids)-args.nval)
    print('train',N,'val',args.nval,flush=True)
    m=Model(args.model).cuda(); m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=0.01)
    import random; rng=random.Random(5)

    def make_seqs(sample_idx, use_neg=True):
        seqs=[]; labels=[]
        for si in sample_idx:
            si=int(si); L=int(ln[si]); n=int(nw[si]); t=int(tg[si])
            st=ws[si]
            # positive at t
            cand=[t]
            negs=set([rng.randint(1,n-1)])
            for c in (t-1,t+1): 
                if 1<=c<=n-1: negs.add(c)
            negs=[g for g in negs if g!=t]
            if use_neg: cand+=negs[:args.nneg]
            for g in cand:
                pos=int(st[g]); seq=ids[si,:L].tolist()
                seq=seq[:pos]+[mask_id]+seq[pos:]
                seqs.append(seq); labels.append(1 if g==t else 0)
        return seqs,labels

    def run_batch(seqs,labels,train=True):
        maxT=max(len(s) for s in seqs)
        inp=torch.full((len(seqs),maxT),pad_id,dtype=torch.long)
        att=torch.zeros((len(seqs),maxT),dtype=torch.long); poss=[]
        for j,s in enumerate(seqs):
            inp[j,:len(s)]=torch.tensor(s); att[j,:len(s)]=1
            poss.append(s.index(mask_id) if mask_id in s else 0)
        inp=inp.cuda(); att=att.cuda()
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            logit=m(inp,att,poss)
        y=torch.tensor(labels,dtype=torch.float32,device='cuda')
        loss=nn.functional.binary_cross_entropy_with_logits(logit.float(),y)
        if train:
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        return loss.item(), logit.detach().float().cpu().numpy()

    # validation: all gaps for nval held-out sentences
    vs=list(range(len(ids)-args.nval,len(ids)))
    def eval_top1():
        m.eval(); corr=0; tot=0
        with torch.no_grad():
            for st0 in range(0,len(vs),200):
                sub=vs[st0:st0+200]
                seqs=[]; meta=[]
                for si in sub:
                    L=int(ln[si]); n=int(nw[si]); st=ws[si]
                    for g in range(1,n):
                        pos=int(st[g]); seq=ids[si,:L].tolist(); seq=seq[:pos]+[mask_id]+seq[pos:]
                        seqs.append(seq); meta.append((si,g))
                # run all shapes in chunks
                scores=[None]*len(seqs)
                for b0 in range(0,len(seqs),512):
                    bb=seqs[b0:b0+512]
                    maxT=max(len(s) for s in bb)
                    inp=torch.full((len(bb),maxT),pad_id,dtype=torch.long); att=torch.zeros((len(bb),maxT),dtype=torch.long); pp=[]
                    for j,s in enumerate(bb):
                        inp[j,:len(s)]=torch.tensor(s); att[j,:len(s)]=1; pp.append(s.index(mask_id))
                    inp=inp.cuda(); att=att.cuda()
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        lg=m(inp,att,pp).float().cpu().numpy()
                    for j,v in enumerate(lg): scores[b0+j]=v
                # group
                cur=[]; cursi=None
                for (si,g),v in zip(meta,scores):
                    if si!=cursi:
                        if cur: 
                            gg,ss=zip(*cur); tot+=1; corr+=(gg[int(np.argmax(ss))]==int(tg[cursi]))
                        cur=[]; cursi=si
                    cur.append((g,v))
                if cur:
                    gg,ss=zip(*cur); tot+=1; corr+=(gg[int(np.argmax(ss))]==int(tg[cursi]))
        m.train(); return corr/tot
    print('init val top1',eval_top1(),flush=True)
    step=0; t0=time.time()
    order=np.arange(N)
    for ep in range(args.epochs):
        np.random.shuffle(order)
        for st in range(0,N,args.batch):
            idx=order[st:st+args.batch]
            seqs,labels=make_seqs(idx)
            loss,_=run_batch(seqs,labels,train=True)
            step+=1
            if step%500==0: print('ep',ep,'step',step,'loss',round(loss,4),'t',round(time.time()-t0),flush=True)
        acc=eval_top1()
        print(f'== ep{ep} val top1={acc:.4f}',flush=True)
        torch.save({'model':m.state_dict(),'model_name':args.model},args.out)
    print('saved',args.out)

if __name__=='__main__': main()
