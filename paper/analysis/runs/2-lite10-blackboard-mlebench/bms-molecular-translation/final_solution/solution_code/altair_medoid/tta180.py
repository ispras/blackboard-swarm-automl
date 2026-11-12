import os,sys,time,argparse,numpy as np,pandas as pd,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from iota_common import decode_ids
from infer_iota import load, to_inchi_parallel
ROOT='/download_job_code/payload/sandbox'

class DS(Dataset):
    def __init__(self,ids,split,rot,size): self.ids=ids; self.split=split; self.rot=rot; self.size=size
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        iid=self.ids[i]; base=f'{ROOT}/data/train' if self.split=='train' else f'{ROOT}/data/test'
        im=Image.open(f'{base}/{iid[0]}/{iid[1]}/{iid[2]}/{iid}.png').convert('L')
        if self.rot: im=im.rotate(self.rot,expand=True,fillcolor=255)
        w,h=im.size; m=max(w,h)
        if (w,h)!=(m,m):
            n=Image.new('L',(m,m),255); n.paste(im,((m-w)//2,(m-h)//2)); im=n
        im=im.resize((self.size,self.size),Image.BILINEAR)
        return torch.from_numpy(np.array(im,dtype=np.uint8)).unsqueeze(0).repeat(3,1,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--ids_csv',required=True)   # has image_id column
    ap.add_argument('--split',default='test',choices=['test','train'])
    ap.add_argument('--out',required=True)
    ap.add_argument('--rot',type=int,default=180)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args()
    ids=pd.read_csv(a.ids_csv).image_id.astype(str).tolist()
    model,ca=load(a.ckpt); size=ca.get('size',256)
    dl=DataLoader(DS(ids,a.split,a.rot,size),batch_size=a.batch,shuffle=False,num_workers=a.workers,pin_memory=True)
    oids=[]; osm=[]; ocf=[]; t0=time.time()
    for bi,x in enumerate(dl):
        x=x.cuda(non_blocking=True).float().div_(255).sub_(0.5).div_(0.5)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _r=model.generate(x,maxlen=160,return_scores=True); ys,c=_r[0],_r[1]
        osm+=[decode_ids(r) for r in ys.cpu().tolist()]; ocf+=c.cpu().tolist()
        oids.extend(ids[len(oids):len(oids)+x.size(0)])
        if (bi+1)%25==0:
            el=time.time()-t0; print(f'{len(osm)}/{len(ids)} {len(osm)/el:.1f} img/s',flush=True)
    inc=to_inchi_parallel(osm,16)
    out=pd.DataFrame({'image_id':oids,'smiles':osm,'conf':ocf,'inchi_from_smiles':inc})
    out['valid']=out.inchi_from_smiles.astype(str).str.len()>0
    out.to_csv(a.out,index=False); print('saved',a.out,out.shape,flush=True)

if __name__=='__main__': main()
