import os, sys, time, csv, argparse, cv2, numpy as np, pandas as pd, torch
sys.path.insert(0,'/download_job_code/payload/sandbox/workspaces/altair_theta')
from molscribe import MolScribe
from molscribe.dataset import CropWhite
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi as rdinchi
RDLogger.DisableLog('rdApp.*')
ROOT='/download_job_code/payload/sandbox'
CKPT='/download_job_code/payload/sandbox/workspaces/altair_theta/hf/models--yujieq--MolScribe/snapshots/a0189776b7415b82795c7ee81eed311bf5c8724b/swin_base_char_aux_1m.pth'

def make_tf(size=384):
    tl=[CropWhite(pad=5),
        A.PadIfNeeded(min_height=size,min_width=size,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255)),
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size,min_width=size,border_mode=cv2.BORDER_CONSTANT,value=(255,255,255)),
        A.ToGray(p=1),A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),ToTensorV2()]
    return A.Compose(tl,keypoint_params=A.KeypointParams(format='xy',remove_invisible=False))

def to_inchi(s):
    try:
        m=Chem.MolFromSmiles(s) if s else None
        if m is None: return ''
        return rdinchi.MolToInchi(m) or ''
    except Exception: return ''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--limit', type=int, default=0)
    a=ap.parse_args()
    ids=pd.read_csv(a.ids).image_id.astype(str).tolist()
    if a.limit: ids=ids[:a.limit]
    paths=[f'{ROOT}/data/test/{i[0]}/{i[1]}/{i[2]}/{i}.png' for i in ids]
    # sort by size for batching, remember order
    sizes=np.array([os.path.getsize(p) if os.path.exists(p) else 0 for p in paths])
    order=np.argsort(sizes)
    model=MolScribe(model_path=CKPT, device=torch.device('cuda'))
    for l in model.encoder.transformer.layers: l.use_checkpoint=False
    model.transform=make_tf(384)
    model.decoder.compute_confidence=True
    out=[]; t0=time.time(); n=0
    for bi in range(0,len(order),a.bs):
        js=order[bi:bi+a.bs]
        imgs=[]
        for j in js:
            im=cv2.imread(paths[j])
            imgs.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB) if im is not None else None)
        valid=[k for k,im in enumerate(imgs) if im is not None]
        b=[imgs[k] for k in valid]
        if not b: continue
        with torch.inference_mode():
            x=torch.stack([model.transform(image=z,keypoints=[])['image'] for z in b]).cuda()
            f,h=model.encoder(x)
            preds=model.decoder.decode(f,h)
        raw=[p['chartok_coords']['smiles'] for p in preds]
        confs=[float(p.get('overall_score', p.get('chartok_coords',{}).get('average_token_score',-1.0))) for p in preds]
        for k,rr,cf in zip(valid,raw,confs):
            j=js[k]
            out.append((ids[j], rr or '', cf))
        n+=len(b)
        if (bi//a.bs)%5==0:
            el=time.time()-t0; print(f'{n}/{len(order)} {n/el:.2f} img/s eta {(len(order)-n)/max(n/el,1e-6)/60:.1f}m',flush=True)
            pd.DataFrame(out,columns=['image_id','smiles','conf']).to_csv(a.out+'.raw.csv',index=False)
    pd.DataFrame(out,columns=['image_id','smiles','conf']).to_csv(a.out+'.raw.csv',index=False)
    from multiprocessing import Pool
    def work(c):
        return [to_inchi(x) for x in c]
    smis=[x[1] for x in out]
    chunks=[smis[i:i+2000] for i in range(0,len(smis),2000)]
    with Pool(12) as pool: res=pool.map(work, chunks)
    inchis=[y for r in res for y in r]
    df=pd.DataFrame({'image_id':[x[0] for x in out],'inchi':inchis,'conf':[x[2] for x in out]})
    df['valid']=df.inchi.astype(str).str.len()>0
    df.to_csv(a.out,index=False)
    print('saved',a.out,len(df),'valid',float(df.valid.mean()),'time',round(time.time()-t0,1),flush=True)
if __name__=='__main__': main()
