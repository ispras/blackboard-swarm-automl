"""Sharded resumable test inference for rigel_mu's ConvNeXt+BPE-transformer model.
Imports rigel's model.py + tokenizer live. Usage (under systemd-run --scope):
  python infer_rigel.py --ckpt workspaces/rigel_mu/ck_v1_best.pt --start 0 --end 121000 --out shard_0.csv [--bs 128]
"""
import sys, os, time, argparse, math
ROOT='/download_job_code/payload/sandbox'
WORK=f'{ROOT}/workspaces/rigel_mu'
sys.path.append('/usr/local/lib/python3.11/site-packages')
sys.path.insert(0, WORK)
import numpy as np, pandas as pd, torch, cv2
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model import Model

IMAGENET_MEAN=np.array([0.485,0.456,0.406],np.float32)
IMAGENET_STD=np.array([0.229,0.224,0.225],np.float32)

class TestDS(Dataset):
    def __init__(self, ids, img_size, imgroot):
        self.ids=ids; self.img_size=img_size; self.imgroot=imgroot
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        iid=self.ids[i]
        p=f'{self.imgroot}/{iid[0]}/{iid[1]}/{iid[2]}/{iid}.png'
        img=cv2.imread(p)
        if img is None: img=np.zeros((self.img_size,self.img_size,3),np.uint8)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(self.img_size,self.img_size),interpolation=cv2.INTER_AREA)
        img=(img.astype(np.float32)/255.0-IMAGENET_MEAN)/IMAGENET_STD
        img=np.transpose(img,(2,0,1))
        return torch.from_numpy(img), i

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',default=f'{WORK}/ck_v1_best.pt')
    ap.add_argument('--start',type=int,default=0); ap.add_argument('--end',type=int,default=-1)
    ap.add_argument('--bs',type=int,default=128); ap.add_argument('--out',required=True)
    ap.add_argument('--img_size',type=int,default=384); ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--max_len',type=int,default=256)
    ap.add_argument('--ids_csv',default='',help='csv with image_id col; default = sample_submission test ids')
    ap.add_argument('--imgroot',default=f'{ROOT}/data/test')
    a=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    torch.set_num_threads(4)
    ck=torch.load(a.ckpt,map_location='cpu'); args=ck.get('args',{})
    tok=Tokenizer.from_file(f'{WORK}/bpe1024.json')
    BOS,EOS=tok.token_to_id('[BOS]'),tok.token_to_id('[EOS]')
    vocab=tok.get_vocab_size()
    model=Model(vocab, enc_name=args.get('enc','convnext_tiny.fb_in22k_ft_in1k'),
                out_index=args.get('out_index',2), d_model=args.get('d_model',512),
                nhead=args.get('nhead',8), layers=args.get('layers',6),
                dim_ff=args.get('dim_ff',2048), dropout=args.get('dropout',0.1),
                max_len=int(ck['model']['tgt_pos.weight'].shape[0]), pretrained=False).cuda()
    model.load_state_dict(ck['model']); model.eval()
    if a.ids_csv:
        ids=pd.read_csv(a.ids_csv).image_id.astype(str).tolist()
    else:
        ids=pd.read_csv(f'{ROOT}/data/sample_submission.csv').image_id.tolist()
    imgroot=a.imgroot
    end=len(ids) if a.end<0 else min(a.end,len(ids)); ids=ids[a.start:end]
    ds=TestDS(ids,a.img_size,imgroot)
    dl=DataLoader(ds,batch_size=a.bs,shuffle=False,num_workers=a.workers,pin_memory=True)
    rows=[]; t0=time.time(); done=0
    for imgs,idx in dl:
        imgs=imgs.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outs=model.generate(imgs,BOS,EOS,max_len=a.max_len)
        for o,i in zip(outs,idx.tolist()):
            rows.append((ids[i], tok.decode(o)))
        done+=len(outs)
        if done % (a.bs*10)==0:
            el=time.time()-t0; print(f'{done}/{len(ids)} {done/el:.2f} img/s eta {(len(ids)-done)/max(done/el,1e-9)/60:.1f}min',flush=True)
            pd.DataFrame(rows,columns=['image_id','InChI']).to_csv(a.out,index=False)
    pd.DataFrame(rows,columns=['image_id','InChI']).to_csv(a.out,index=False)
    print('done',len(rows),'in',time.time()-t0,'s ->',a.out)
if __name__=='__main__': main()
