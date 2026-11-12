import argparse, json, numpy as np, torch, torch.nn as nn
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
        return self.head(h[idx,torch.tensor(poss,device=ids.device)]).squeeze(-1)

def load_rows(args):
    rows=[]
    if args.test_csv:
        import csv
        with open(args.test_csv,newline='',encoding='utf-8') as f:
            r=csv.reader(f); next(r)
            for row in r: rows.append((int(row[0]),row[1]))
    else:
        for l in open(args.input):
            d=json.loads(l); rows.append((int(d['id']), d.get('corrupted') or d.get('sentence')))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='roberta-base')
    ap.add_argument('--ckpt',default='verif_ft.pt')
    ap.add_argument('--input',default='')
    ap.add_argument('--test_csv',default='')
    ap.add_argument('--out',default='verif_ft_gaps.jsonl')
    ap.add_argument('--chunk',type=int,default=2000)
    ap.add_argument('--batch',type=int,default=768)
    ap.add_argument('--maxtok',type=int,default=128)
    ap.add_argument('--topk',type=int,default=10)
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model, **({"add_prefix_space":True} if ("roberta" in args.model or "gpt2" in args.model) else {}))
    model=Model(args.model)
    sd=torch.load(args.ckpt,map_location='cpu')
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    model.eval().cuda(); mask_id=tok.mask_token_id; pad_id=tok.pad_token_id or 1
    rows=load_rows(args); print('rows',len(rows),flush=True)
    out=open(args.out,'w'); done=0
    def tok_sent(sent):
        words=sent.split(' ')
        if len(words)<3: return None
        enc=tok(words,is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=args.maxtok)
        ids=enc['input_ids']; wids=enc.word_ids()
        first={}
        for ti,w in enumerate(wids):
            if w is not None and w not in first: first[w]=ti
        nw=len(first)
        if nw<3 or len(first)!=len(words): return None
        return ids,[first[g] for g in range(nw)],nw
    with torch.no_grad():
        for cst in range(0,len(rows),args.chunk):
            chunk=rows[cst:cst+args.chunk]
            items=[]; keep=[]
            for rid,sent in chunk:
                t=tok_sent(sent)
                if t is None:
                    out.write(json.dumps({'id':rid,'gap_probs':[]})+'\n'); continue
                ids,starts,nw=t; keep.append((rid,nw))
                for g in range(1,nw): items.append((ids,starts[g]))
            allsc=[]
            for bi in range(0,len(items),args.batch):
                batch=items[bi:bi+args.batch]
                maxT=max(len(ids)+1 for ids,_ in batch)
                inp=torch.full((len(batch),maxT),pad_id,dtype=torch.long); att=torch.zeros((len(batch),maxT),dtype=torch.long); poss=[]
                for j,(ids,pos) in enumerate(batch):
                    seq=ids[:pos]+[mask_id]+ids[pos:]
                    inp[j,:len(seq)]=torch.tensor(seq); att[j,:len(seq)]=1; poss.append(pos)
                inp=inp.cuda(); att=att.cuda()
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    sc=model(inp,att,poss).float()
                allsc.append(sc.cpu().numpy())
            scores=np.concatenate(allsc) if allsc else np.array([])
            k=0
            for rid,nw in keep:
                s=scores[k:k+nw-1]; k+=nw-1
                ss=torch.tensor(s); prob=torch.softmax(ss,dim=-1).numpy()
                kk=min(args.topk,len(s)); top=np.argsort(-s)[:kk]
                gp=[[int(i)+1,float(s[i]),float(prob[i])] for i in top]
                out.write(json.dumps({'id':rid,'gap_probs':gp})+'\n'); done+=1
            if (cst//args.chunk)%10==0: print('done',done,'/',len(rows),flush=True)
    out.close(); print('wrote',args.out,done)

if __name__=='__main__': main()
