import pickle,sys,argparse
import numpy as np
from Levenshtein import distance as lev
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
sys.path.insert(0,'scripts')
from word_at_oracle import build_vocab

def cw(words,probs,lam=0.2):
    S=sum(probs) or 1; pf=[p/S for p in probs]; best=None;bu=None
    for u in words:
        e=sum(pf[i]*lev(u,words[i]) for i in range(len(words)))+lam*len(u)
        if best is None or e<best: best=e;bu=u
    return bu
def ins(t,g,w): return ' '.join(t[:g])+' '+str(w)+' '+' '.join(t[g:])

def word_fuse(wd,pd,wm,pm,a,b,lam=0.2,eps=1e-6):
    d={}; 
    for w,p in zip(wd,pd):
        d[w]=[a*np.log(max(p,eps)), b*np.log(eps)]
    for w,p in zip(wm,pm):
        if w in d: d[w][1]=b*np.log(max(p,eps))
        else: d[w]=[a*np.log(eps), b*np.log(max(p,eps))]
    items=[(w,v[0]+v[1]) for w,v in d.items()]
    items.sort(key=lambda x:-x[1])
    items=items[:25]
    ws=[w for w,s in items]; mx=max(s for w,s in items)
    ex=np.exp(np.array([s for w,s in items])-mx); ps=(ex/ex.sum()).tolist()
    return ws,ps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dump',required=True)
    ap.add_argument('--tri',default='/download_job_code/payload/sandbox/workspaces/spica_psi/tri_holdout_allgaps.pkl')
    ap.add_argument('--mlm',default='mlm_allgaps_1500.pkl')
    ap.add_argument('--signals',default='det+tri+mlm')
    ap.add_argument('--lam',type=float,default=0.2)
    ap.add_argument('--wa',type=float,default=0.0)
    ap.add_argument('--wb',type=float,default=1.0)
    ap.add_argument('--wlam',type=float,default=0.0)
    ap.add_argument('--dlam',type=float,default=0.2)
    ap.add_argument('--wl_ckpt',default='')
    ap.add_argument('--wl_bs',type=int,default=64)
    ap.add_argument('--wl_topk',type=int,default=20)
    ap.add_argument('--gpu_frac',type=float,default=0.15)
    ap.add_argument('--meta',default='')
    ap.add_argument('--orig',default='')
    ap.add_argument('--save_cache',default='')
    args=ap.parse_args()
    D=pickle.load(open(args.dump,'rb')); toks=D['tokens']
    if 'res' in D:
        res=D['res']; meta=D['meta']; orig=D['orig']
    else:
        res=[{'gap_probs':D['gap_probs'][i],'words':[],'probs':[],'trunc':D['trunc'][i],'detwp':D['detwp'][i]} for i in range(len(toks))]
        meta=[l.rstrip('\n').split('\t') for l in open(args.meta)][1:]
        orig=[l.rstrip('\n') for l in open(args.orig)]
    TRI=pickle.load(open(args.tri,'rb')) if args.tri and args.tri!='none' else []
    MLM=pickle.load(open(args.mlm,'rb')) if args.mlm and args.mlm!='none' else [{}]*len(toks)
    N=len(toks)
    use=args.signals.split('+')
    feats=[None]*N; labels=[None]*N; valid=[False]*N
    for i in range(N):
        t=toks[i]; n=len(t); r=res[i]; gp=r['gap_probs']
        if len(gp)==0 or len(gp)!=n-1: continue
        ng=n-1
        tri=TRI[i] if i<len(TRI) else {}
        tg=np.full(ng,-30.0); tp=np.zeros(ng)
        for k,(w,gain,ls) in tri.items():
            if 1<=k<=ng: tg[k-1]=gain; tp[k-1]=1.0
        mm=np.zeros(ng); m2=np.zeros(ng)
        for g in range(1,ng+1):
            d=MLM[i].get(g)
            if d:
                mm[g-1]=d['maxp']; m2[g-1]=d['p'][1] if len(d['p'])>1 else 0.0
        detwp=np.array(r['detwp']) if r.get('detwp') is not None else np.array([r['probs'][g][0] if r['probs'][g] else 0.0 for g in range(len(gp))])
        detrank=np.argsort(np.argsort(-gp)); trirank=np.argsort(np.argsort(-tg)); mlmrank=np.argsort(np.argsort(-mm))
        frac=np.arange(1,ng+1)/n
        gpr=np.clip(gp,1e-6,1)
        cols={'det':[gpr,np.log(gpr),-detrank,detwp],
              'tri':[np.clip(tg,-30,5),tp,-trirank],
              'mlm':[mm,np.log(np.clip(mm,1e-6,1)),-mlmrank,mm-np.clip(m2,0,1)]}
        X=np.column_stack([frac]+[c for s in use for c in cols[s]])
        y=(np.arange(1,ng+1)==int(meta[i][2])).astype(int)
        feats[i]=X; labels[i]=y; valid[i]=True
    idxs=[i for i in range(N) if valid[i]]
    Xall=np.vstack([feats[i] for i in idxs]); yall=np.hstack([labels[i] for i in idxs])
    gall=np.repeat(np.array(idxs),[len(labels[i]) for i in idxs])
    local=np.concatenate([np.arange(len(labels[i])) for i in idxs])
    gkf=GroupKFold(n_splits=3); out={}
    for tr,te in gkf.split(Xall,yall,gall):
        sc=StandardScaler().fit(Xall[tr]); clf=LogisticRegression(max_iter=3000,C=1.0).fit(sc.transform(Xall[tr]),yall[tr])
        pr=clf.predict_proba(sc.transform(Xall[te]))[:,1]
        for j,gi in enumerate(te):
            s_=gall[gi]; out.setdefault(s_,{})[int(local[gi])]=pr[j]
    post={}
    for s_ in idxs:
        arr=np.zeros(len(feats[s_]))
        for gi,v in out[s_].items(): arr[gi]=v
        post[s_]=arr/arr.sum()
    WL={}
    if args.wl_ckpt:
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac,0)
        wtok=AutoTokenizer.from_pretrained(args.wl_ckpt)
        wmodel=AutoModelForMaskedLM.from_pretrained(args.wl_ckpt,torch_dtype=torch.float16).cuda().eval()
        ids_all,id2w=build_vocab(wtok); ids_t=torch.tensor(ids_all,device='cuda')
        pairs=[]
        for i in range(N):
            if valid[i]:
                gs=int(post[i].argmax())+1; pairs.append((i,gs))
        texts=[' '.join(toks[i][:gs]+[wtok.mask_token]+toks[i][gs:]) for i,gs in pairs]
        for s0 in range(0,len(texts),args.wl_bs):
            chunk=texts[s0:s0+args.wl_bs]
            enc=wtok(chunk,padding=True,truncation=True,max_length=200,return_tensors='pt')
            ii=enc['input_ids'].cuda(); am=enc['attention_mask'].cuda()
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
                lo=wmodel(input_ids=ii,attention_mask=am).logits
            mp=(ii==wtok.mask_token_id); idx=mp.float().argmax(1); ar=torch.arange(len(chunk),device='cuda')
            pr=torch.softmax(torch.log_softmax(lo[ar,idx].float(),-1)[:,ids_t],-1)
            k=min(args.wl_topk,pr.size(1)); tv,ti=torch.topk(pr,k,-1); tv=tv.cpu().numpy(); ti=ti.cpu().numpy()
            for r in range(len(chunk)):
                i,gs=pairs[s0+r]
                WL[(i,gs)]=([id2w[int(ids_all[ti[r,j]])] for j in range(k)],[float(tv[r,j]) for j in range(k)])
        del wmodel; torch.cuda.empty_cache()
        print('wl computed',len(WL),flush=True)
    if args.save_cache:
        pickle.dump({'post':post,'WL':WL,'valid':valid,'tokens':toks,'meta':meta,'orig':orig},open(args.save_cache,'wb'),protocol=4); print('cache saved',args.save_cache,flush=True)
    base=[lev(orig[i],' '.join(toks[i])) for i in range(N)]
    print('baseline',round(np.mean(base),4),'signals',args.signals,'word',args.wa,args.wb)
    resall=[]
    for tau in [0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
        tot=0;app=0;lok=0;wok=0
        for i in range(N):
            if not valid[i]: tot+=base[i]; continue
            q=post[i]; gs=int(q.argmax()); pb=float(q[gs])
            if pb<tau: tot+=base[i]; continue
            g=gs+1; r=res[i]
            wd=r['words'][gs] if r['words'] else []
            pd=r['probs'][gs] if r['words'] else []
            if (i,g) in WL:
                wm,pm=WL[(i,g)]
            else:
                md=MLM[i].get(g)
                if md is None:
                    if not wd: tot+=base[i]; continue
                    wm,pm=wd,pd
                else: wm,pm=md['w'],md['p']
            if args.wa==0 and args.wb==1:
                u=cw(wm,pm,args.wlam)
            elif args.wa==1 and args.wb==0:
                if not wd: tot+=base[i]; continue
                u=cw(wd,pd,args.dlam)
            else:
                ws,ps=word_fuse(wd,pd,wm,pm,args.wa,args.wb); u=cw(ws,ps,args.wlam)
            if u is None: tot+=base[i]; continue
            tot+=lev(ins(toks[i],g,u),orig[i]); app+=1; lok+=(g==int(meta[i][2])); wok+=(u==meta[i][3])
        m=tot/N; resall.append((m,tau,app/N,lok/max(app,1),wok/max(app,1)))
        print(f'tau {tau:.2f} apply {app/N:.3f} metric {m:.4f} locprec {lok/max(app,1):.3f} wordacc {wok/max(app,1):.3f}')
    best=min(resall); print('BEST metric %.4f tau %.2f apply %.3f loc %.3f word %.3f'%best)
if __name__=='__main__': main()
