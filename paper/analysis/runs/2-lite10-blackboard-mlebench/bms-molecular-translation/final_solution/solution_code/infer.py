import os, sys, time, argparse
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0,'/download_job_code/payload/sandbox/workspaces/regulus_iota')
from common import *

ROOT='/download_job_code/payload/sandbox'

def mp_smiles(args):
    smis, = args
    return batch_smiles_to_inchi(smis, load_fallback())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--split', default='test', choices=['val','test'])
    p.add_argument('--out', required=True)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--maxlen', type=int, default=160)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--ids_csv', type=str, default='')
    p.add_argument('--rot180', action='store_true')
    args=p.parse_args()
    ck=torch.load(args.ckpt, map_location='cpu')
    ca=ck['args']
    size=ca.get('size',256)
    model=Seq2Seq(ca.get('d_model',512), ca.get('nheads',8), ca.get('nlayers',6), ca.get('ffn',2048), ca.get('dropout',0.1), pretrained=False)
    model.load_state_dict(ck['model'])
    model=model.cuda().eval()
    if args.split=='val':
        val_ids=pd.read_csv(f'{ROOT}/workspaces/regulus_iota/val_20k.csv')[['image_id']]
        lab=pd.read_csv(f'{ROOT}/workspaces/regulus_iota/labels_smiles.csv')
        df=val_ids.merge(lab, on='image_id', how='left')
        img_dir=f'{ROOT}/data/train'
    else:
        df=pd.read_csv(f'{ROOT}/data/sample_submission.csv')[['image_id']].copy()
        df['smiles']='C'; df['InChI']=''
        img_dir=f'{ROOT}/data/test'
    if args.split=='test' and args.ids_csv:
        order=pd.read_csv(args.ids_csv)
        if 'size' in order.columns:
            order=order.sort_values('size').reset_index(drop=True)
        base=df.copy()
        df=order[['image_id']].merge(base, on='image_id', how='left')
        if df.smiles.isna().any():
            miss=base[~base.image_id.isin(set(df.image_id))]
            df=pd.concat([df, miss], ignore_index=True)
        df=df.reset_index(drop=True)
    if args.limit: df=df.iloc[:args.limit].reset_index(drop=True)
    if args.nshard>1:
        df=df.iloc[args.shard::args.nshard].reset_index(drop=True)
    print('rows',len(df),flush=True)
    ds=InChIDataset(df, img_dir, size, train=False)
    dl=DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate, pin_memory=True)
    all_ids=[]; all_smi=[]; all_conf=[]; all_minconf=[]
    t0=time.time(); n=0
    for bi,(x,t) in enumerate(dl):
        x=x.cuda(non_blocking=True); x=x.float().div_(255).sub_(0.5).div_(0.5)
        if args.rot180: x=torch.flip(x,dims=[2,3])
        ys,conf,minconf=model.generate(x, maxlen=args.maxlen, return_scores=True)
        for row,c,mc in zip(ys.cpu().tolist(), conf.cpu().tolist(), minconf.cpu().tolist()):
            all_smi.append(decode_ids(row)); all_conf.append(c); all_minconf.append(mc)
        n+=len(all_smi)-len(all_smi)  # noop
        all_ids.extend(df.image_id.values[len(all_ids):len(all_ids)+x.size(0)].tolist())
        if (bi+1)%50==0:
            done=len(all_smi); el=time.time()-t0
            print(f'{done}/{len(df)} {done/el:.1f} img/s elapsed {el/60:.1f}m',flush=True)
    assert len(all_ids)==len(all_smi), (len(all_ids),len(all_smi))
    print('decoded',len(all_smi),'time',round(time.time()-t0,1),flush=True)
    # parallel smiles->inchi
    from multiprocessing import Pool
    t=time.time()
    chunks=[all_smi[i:i+5000] for i in range(0,len(all_smi),5000)]
    with Pool(16) as pool:
        res=pool.map(mp_smiles, [(c,) for c in chunks])
    inchis=[x for r in res for x in r]
    print('converted inchi',len(inchis),'time',round(time.time()-t,1),flush=True)
    out=pd.DataFrame({'image_id':all_ids,'smiles':all_smi,'conf':all_conf,'minconf':all_minconf,'inchi_from_smiles':inchis})
    FB=load_fallback()
    out['valid']=out.inchi_from_smiles.ne('') & (out.inchi_from_smiles!=FB)
    out.loc[~out['valid'],'inchi_from_smiles']=''
    out.to_csv(args.out, index=False)
    print('saved',args.out,out.shape,flush=True)
    if args.split=='val':
        from rapidfuzz.distance import Levenshtein as Lev
        fb=load_fallback()
        gt=df.InChI.tolist()
        for th in [-2.0,-1.5,-1.2,-1.0,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.25,-0.2]:
            preds=[(r.inchi_from_smiles if (r.conf>=th and r.smiles!='') else fb) for r in out.itertuples()]
            lev=np.mean([Lev.distance(a,b) for a,b in zip(preds,gt)])
            print(f'threshold {th}: lev {lev:.4f}',flush=True)

if __name__=='__main__':
    main()
