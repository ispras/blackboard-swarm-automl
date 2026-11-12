import os, sys, time, argparse, csv, cv2, numpy as np, pandas as pd, torch
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi as rdinchi
RDLogger.DisableLog('rdApp.*')
from molscribe import MolScribe
from molscribe.dataset import CropWhite
from molscribe.chemistry import convert_graph_to_smiles
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT='/download_job_code/payload/sandbox'
CKPT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'hf/models--yujieq--MolScribe/snapshots/a0189776b7415b82795c7ee81eed311bf5c8724b/swin_base_char_aux_1m.pth')

def make_tf(size=384):
    tl=[CropWhite(pad=5),
        A.PadIfNeeded(min_height=size,min_width=size,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255)),
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size,min_width=size,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255)),
        A.ToGray(p=1),A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),ToTensorV2()]
    return A.Compose(tl,keypoint_params=A.KeypointParams(format='xy',remove_invisible=False))

def build_model(dev):
    m=MolScribe(model_path=CKPT, device=dev)
    for l in m.encoder.transformer.layers: l.use_checkpoint=False
    m.transform=make_tf(384)
    m.decoder.compute_confidence=True
    return m

def to_inchi(smi):
    if not smi: return ''
    try:
        mol=Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None: return ''
        return rdinchi.MolToInchi(mol) or ''
    except Exception:
        return ''

def predict_arrays(model, imgs, bs=128, dev='cuda'):
    preds=[]
    with torch.inference_mode():
        for i in range(0,len(imgs),bs):
            b=imgs[i:i+bs]
            x=torch.stack([model.transform(image=z,keypoints=[])['image'] for z in b]).to(dev)
            f,h=model.encoder(x)
            preds+=model.decoder.decode(f,h)
    raw=[p['chartok_coords']['smiles'] for p in preds]
    coords=[p['chartok_coords']['coords'] for p in preds]
    syms=[p['chartok_coords']['symbols'] for p in preds]
    edges=[p['edges'] for p in preds]
    smi_graph,_,_=convert_graph_to_smiles(coords,syms,edges,images=imgs)
    conf=[p.get('overall_score',-1.0) for p in preds]
    return smi_graph, raw, conf

def load_rgb(p):
    im=cv2.imread(p)
    if im is None: return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--val_csv',default='')
    ap.add_argument('--out',required=True)
    ap.add_argument('--bs',type=int,default=128)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--nshards',type=int,default=1)
    ap.add_argument('--sort',default='size')
    ap.add_argument('--log_every',type=int,default=20)
    args=ap.parse_args()
    dev=torch.device('cuda')

    if args.val_csv:
        df=pd.read_csv(args.val_csv)
        if args.limit: df=df.iloc[:args.limit]
        ids=df.image_id.tolist(); paths=[f"{ROOT}/data/train/{i[0]}/{i[1]}/{i[2]}/{i}.png" for i in ids]
    else:
        df=pd.read_csv(f'{ROOT}/data/sample_submission.csv')
        ids=df.image_id.tolist()
        paths=[f"{ROOT}/data/test/{i[0]}/{i[1]}/{i[2]}/{i}.png" for i in ids]

    # order: sort by file size descending/ascending for batch-length homogeneity
    if args.sort in ('size','area'):
        if args.sort=='size':
            key=np.array([os.path.getsize(p) if os.path.exists(p) else 0 for p in paths])
        else:
            key=[]
            from PIL import Image
            for p in paths:
                try: w,h=Image.open(p).size; key.append(w*h)
                except Exception: key.append(0)
            key=np.array(key)
        order=np.argsort(key)
    else:
        order=np.arange(len(paths))
    # shard round-robin on sorted order, skip already done
    done=set()
    if os.path.exists(args.out):
        try:
            prev=pd.read_csv(args.out, usecols=['image_id'])
            done=set(prev.image_id.astype(str))
            print(f"resume: {len(done)} done", flush=True)
        except Exception: pass
    sel=[int(j) for k,j in enumerate(order) if k % args.nshards == args.shard]
    sel=[j for j in sel if ids[j] not in done]
    if args.start: sel=sel[args.start:]
    if args.limit: sel=sel[:args.limit]
    print(f"to do: {len(sel)} bs={args.bs} sort={args.sort}", flush=True)

    model=build_model(dev)
    write_header=not os.path.exists(args.out) or os.path.getsize(args.out)==0
    f=open(args.out,'a',newline=''); w=csv.writer(f)
    if write_header: w.writerow(['image_id','smiles_graph','smiles_raw','InChI_pred','confidence','source'])
    t0=time.time(); n=0; nexact=0
    for bi in range(0,len(sel),args.bs):
        js=sel[bi:bi+args.bs]
        imgs=[]; kept=[]
        for j in js:
            im=load_rgb(paths[j])
            if im is None: continue
            imgs.append(im); kept.append(j)
        if not imgs: continue
        sg,sr,conf=predict_arrays(model,imgs,bs=args.bs,dev=dev)
        rows=[]
        for j,g,r,c in zip(kept,sg,sr,conf):
            pred=to_inchi(g); src='graph'
            if not pred:
                pred=to_inchi(r); src='raw' if pred else 'none'
            rows.append([ids[j], g or '', r or '', pred, float(c), src])
        w.writerows(rows); f.flush(); n+=len(rows)
        el=time.time()-t0
        if (bi//args.bs) % args.log_every==0:
            print(f"{n}/{len(sel)} imgs, {el:.0f}s, {el/max(n,1):.4f} s/img, eta {(len(sel)-n)*el/max(n,1)/3600:.2f}h", flush=True)
    f.close()
    print(f"DONE {n} in {(time.time()-t0)/3600:.2f}h", flush=True)

if __name__=='__main__':
    main()
