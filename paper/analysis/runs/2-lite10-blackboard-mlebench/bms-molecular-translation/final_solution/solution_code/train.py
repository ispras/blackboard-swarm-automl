import os, sys, time, math, json, argparse, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0,'/download_job_code/payload/sandbox/workspaces/regulus_iota')
from common import *

ROOT='/download_job_code/payload/sandbox'

def get_args():
    p=argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--val_every', type=int, default=2000)
    p.add_argument('--val_decode', type=int, default=400)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--out', type=str, default='ckpt')
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--nlayers', type=int, default=6)
    p.add_argument('--nheads', type=int, default=8)
    p.add_argument('--ffn', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--time_limit', type=float, default=0.0)  # seconds
    p.add_argument('--maxlen', type=int, default=160)
    p.add_argument('--sched_start', type=int, default=0)
    return p.parse_args()

def make_loaders(args):
    lab=pd.read_csv(f'{ROOT}/workspaces/regulus_iota/labels_smiles.csv')
    val_ids=pd.read_csv(f'{ROOT}/workspaces/regulus_iota/val_20k.csv')[['image_id']]
    val=val_ids.merge(lab, on='image_id', how='left')
    tr=lab[~lab.image_id.isin(set(val_ids.image_id))].reset_index(drop=True)
    if args.limit: tr=tr.iloc[:args.limit].reset_index(drop=True)
    print(f'train rows {len(tr)} val {len(val)}', flush=True)
    dtr=InChIDataset(tr, f'{ROOT}/data/train', args.size, train=True)
    dva=InChIDataset(val, f'{ROOT}/data/train', args.size, train=False)
    ltr=DataLoader(dtr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                   collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=args.workers>0,
                   prefetch_factor=2 if args.workers>0 else None)
    lva=DataLoader(dva, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
    return ltr, lva, val.InChI.tolist()

from rapidfuzz.distance import Levenshtein as Lev

@torch.no_grad()
def evaluate(model, lva, gt_inchis, n_decode=400, max_batches=20, device='cuda', maxlen=160):
    model.eval()
    tok_correct=tok_total=0
    for i,b in enumerate(lva):
        if i>=max_batches: break
        x,t=b; x=x.to(device,non_blocking=True); t=t.to(device,non_blocking=True)
        x=x.float().div_(255).sub_(0.5).div_(0.5)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits=model(x,t[:,:-1])
        pred=logits.argmax(-1); tgt=t[:,1:]
        m=tgt!=PAD
        tok_correct+=(pred[m]==tgt[m]).sum().item(); tok_total+=m.sum().item()
    preds=[]; cnt=0
    for b in lva:
        x,t=b; x=x.to(device,non_blocking=True)
        x=x.float().div_(255).sub_(0.5).div_(0.5)
        ys=model.generate(x, maxlen=maxlen)
        for row in ys.cpu().tolist():
            preds.append(decode_ids(row)); cnt+=1
            if cnt>=n_decode: break
        if cnt>=n_decode: break
    gts=gt_inchis[:len(preds)]
    pin=batch_smiles_to_inchi(preds, load_fallback())
    levs=[Lev.distance(p,g) for p,g in zip(pin,gts)]
    exact_smi=np.mean([p==decode_ids(t.tolist()) for p,t in zip(preds, [])]) if False else 0.0
    model.train()
    return tok_correct/max(tok_total,1), float(np.mean(levs)), 0.0

def main():
    args=get_args()
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    ltr,lva,gt_inchis=make_loaders(args)
    device='cuda'
    model=Seq2Seq(args.d_model, args.nheads, args.nlayers, args.ffn, args.dropout, pretrained=True).to(device)
    nparam=sum(p.numel() for p in model.parameters())
    print('params', nparam, flush=True)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9,0.98))
    step=0; best=1e9
    if args.resume:
        ck=torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); step=ck.get('step',0)
        best=ck.get('best',1e9)
        print('resumed from', args.resume, 'step', step, flush=True)
    total_steps=args.max_steps if args.max_steps>0 else args.epochs*len(ltr)
    def set_lr(s):
        ss=args.sched_start
        if s<ss+args.warmup: lr=args.lr*max(1,(s-ss))/max(1,args.warmup)
        else:
            p=min(1.0,(s-ss-args.warmup)/max(1,total_steps-ss-args.warmup))
            lr=1e-5+(args.lr-1e-5)*0.5*(1+math.cos(math.pi*p))
        for g in opt.param_groups: g['lr']=lr
        return lr
    crit=nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    logpath=os.path.join(args.out,'train_log.csv')
    if not os.path.exists(logpath):
        open(logpath,'w').write('step,tok_acc,lev,exact,time\n')
    logf=open(logpath,'a')
    t0=time.time(); run_loss=0.0; nb=0; tstart=time.time()
    def do_val(reason=''):
        nonlocal best,t0
        ta,lev,ex=evaluate(model,lva,gt_inchis,args.val_decode,maxlen=args.maxlen)
        print(f'VAL step {step} tok_acc {ta:.4f} lev {lev:.4f} {reason}', flush=True)
        logf.write(f'{step},{ta},{lev},{ex},{time.time()}\n'); logf.flush()
        torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'step':step,'args':vars(args),'best':best}, os.path.join(args.out,'ckpt_last.pt'))
        if lev<best:
            best=lev
            torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'step':step,'args':vars(args),'best':best}, os.path.join(args.out,'ckpt_best.pt'))
            print('saved best', flush=True)
        t0=time.time()
    model.train()
    stop=False
    while step < total_steps and not stop:
        for x,t in ltr:
            if step>=total_steps: break
            if args.time_limit>0 and time.time()-tstart>args.time_limit:
                print('time limit reached, stopping', flush=True); stop=True; break
            lr=set_lr(step)
            x=x.to(device,non_blocking=True); t=t.to(device,non_blocking=True)
            x=x.float().div_(255).sub_(0.5).div_(0.5)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits=model(x,t[:,:-1])
                loss=crit(logits.reshape(-1,VOCAB).float(), t[:,1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step+=1; run_loss+=loss.item(); nb+=1
            if step%50==0:
                dt=time.time()-t0
                print(f'step {step}/{total_steps} loss {run_loss/nb:.4f} lr {lr:.2e} {nb/dt:.2f} it/s elapsed {(time.time()-tstart)/60:.1f}m', flush=True)
                run_loss=0; nb=0; t0=time.time()
            if step%args.val_every==0:
                do_val()
    do_val('final')
    print('BEST', best, flush=True)

if __name__=='__main__':
    main()
