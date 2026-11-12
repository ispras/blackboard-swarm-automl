import os, sys, time, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iota_common import Seq2Seq, decode_ids, encode_text, PAD, SOS, EOS, SMILES_CHARS, STOI, MAXLEN
from PIL import Image
ROOT='/download_job_code/payload/sandbox'
FB=open(f'{ROOT}/workspaces/regulus_iota/baseline_constant.txt').read().strip()

class DS(Dataset):
    def __init__(self, ids, img_dir, size):
        self.ids=ids; self.img_dir=img_dir; self.size=size
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        iid=self.ids[i]
        p=os.path.join(self.img_dir, iid[0], iid[1], iid[2], iid+'.png')
        im=Image.open(p).convert('L'); w,h=im.size; m=max(w,h)
        if (w,h)!=(m,m):
            n=Image.new('L',(m,m),255); n.paste(im,((m-w)//2,(m-h)//2)); im=n
        im=im.resize((self.size,self.size), Image.BILINEAR)
        x=torch.from_numpy(np.array(im,dtype=np.uint8)).unsqueeze(0).repeat(3,1,1)
        return x

def load(ckpt, dev='cuda'):
    ck=torch.load(ckpt, map_location='cpu')
    ca=ck['args']
    m=Seq2Seq(ca.get('d_model',512), ca.get('nheads',8), ca.get('nlayers',6), ca.get('ffn',2048), ca.get('dropout',0.1), pretrained=False)
    m.load_state_dict(ck['model']); m=m.to(dev).eval()
    return m, ca

def _smiles_worker(c):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import inchi
    RDLogger.DisableLog('rdApp.*')
    out=[]
    for s in c:
        try:
            mol=Chem.MolFromSmiles(s) if s else None
            out.append(inchi.MolToInchi(mol) if mol is not None else '')
        except Exception: out.append('')
    return out

def to_inchi_parallel(smiles, workers=16):
    from multiprocessing import Pool
    chunks=[smiles[i:i+3000] for i in range(0,len(smiles),3000)]
    with Pool(workers) as p: res=p.map(_smiles_worker, chunks)
    return [x for r in res for x in r]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--val',action='store_true')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--workers',type=int,default=10)
    ap.add_argument('--maxlen',type=int,default=160)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    dev='cuda'
    model,ca=load(a.ckpt,dev); size=ca.get('size',256)
    if a.val:
        ids=pd.read_csv(f'{ROOT}/workspaces/regulus_iota/val_20k.csv').image_id.tolist(); img_dir=f'{ROOT}/data/train'
    else:
        ids=pd.read_csv(f'{ROOT}/workspaces/altair_theta/test_ids_sorted.csv').image_id.astype(str).tolist(); img_dir=f'{ROOT}/data/test'
    if a.limit: ids=ids[:a.limit]
    raw_out=a.out+'.raw.csv'
    done=set()
    if a.resume and os.path.exists(raw_out):
        try: done=set(pd.read_csv(raw_out,usecols=['image_id']).image_id.astype(str)); print('resume',len(done),flush=True)
        except Exception: pass
    ids=[i for i in ids if i not in done]
    print(f"rows {len(ids)} size {size}",flush=True)
    dl=DataLoader(DS(ids,img_dir,size),batch_size=a.batch,shuffle=False,num_workers=a.workers,pin_memory=True)
    oids=[]; osm=[]; ocf=[]
    def flush():
        if not oids: return
        df=pd.DataFrame({'image_id':oids,'smiles':osm,'conf':ocf})
        hdr = not os.path.exists(raw_out) or os.path.getsize(raw_out)==0
        df.to_csv(raw_out, mode='a', header=hdr, index=False)
        oids.clear(); osm.clear(); ocf.clear()
    t0=time.time(); seen=0
    for bi,x in enumerate(dl):
        x=x.to(dev,non_blocking=True).float().div_(255).sub_(0.5).div_(0.5)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _r=model.generate(x,maxlen=a.maxlen,return_scores=True); ys,conf=_r[0],_r[1]
        for row,c in zip(ys.cpu().tolist(),conf.cpu().tolist()):
            osm.append(decode_ids(row)); ocf.append(float(c))
        oids.extend(ids[seen:seen+x.size(0)]); seen+=x.size(0)
        if (bi+1)%25==0:
            flush(); el=time.time()-t0
            print(f"{seen}/{len(ids)} {seen/el:.1f} img/s eta {(len(ids)-seen)/max(seen/el,1e-6)/3600:.2f}h",flush=True)
    flush()
    print('decoded',len(ids),'time',round(time.time()-t0,1),flush=True)
    raw=pd.read_csv(raw_out)
    inchis=to_inchi_parallel(raw.smiles.fillna('').astype(str).tolist(), workers=16)
    out=pd.DataFrame({'image_id':raw.image_id,'smiles':raw.smiles,'conf':raw.conf,'inchi_from_smiles':inchis})
    out['valid']=out.inchi_from_smiles.astype(str).str.len()>0
    out.to_csv(a.out,index=False)
    print('saved',a.out,out.shape,flush=True)
    if a.val:
        from rapidfuzz.distance import Levenshtein as Lev
        v=pd.read_csv(f'{ROOT}/workspaces/regulus_iota/val_20k.csv'); gt=dict(zip(v.image_id,v.InChI))
        for th in [-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0]:
            preds=[(x if (c>=th and x) else FB) for x,c in zip(out.inchi_from_smiles,out.conf)]
            d=[Lev.distance(p,gt[i]) for p,i in zip(preds,out.image_id) if i in gt]
            print(f"th {th:+.2f} lev {np.mean(d):.3f} exact {np.mean(np.array(d)==0):.3f}",flush=True)

if __name__=='__main__': main()
