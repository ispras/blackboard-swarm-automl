import argparse, os, json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import timm
from torchvision.transforms import functional as TF
import train as T

DATA=T.DATA

def make_loader(df, img_dir, size, bs, workers):
    ds=T.CassavaDS(df,img_dir,size,train=False)
    return DataLoader(ds,batch_size=bs,shuffle=False,num_workers=workers,pin_memory=True)

@torch.no_grad()
def infer(model, df, img_dir, size, bs, workers, flips=(0,), scales=(1.0,)):
    acc=None; n=0
    for sc in scales:
        s=int(round(size*sc)); dl=make_loader(df,img_dir,s,bs,workers); out=[]
        for x,_ in dl:
            x=x.cuda(non_blocking=True); pr=0
            with torch.autocast('cuda',dtype=torch.float16):
                for f in flips:
                    xx=x
                    if f&1: xx=torch.flip(xx,dims=[3])
                    if f&2: xx=torch.flip(xx,dims=[2])
                    pr=pr+torch.softmax(model(xx).float(),1)
            out.append((pr/len(flips)).cpu().numpy())
        p=np.concatenate(out); acc=p if acc is None else acc+p; n+=1
    return acc/n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True); ap.add_argument('--ckpt',default='best.pth')
    ap.add_argument('--bs',type=int,default=24); ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--test',action='store_true'); ap.add_argument('--outdir',default=None)
    ap.add_argument('--flips-only',action='store_true')
    args=ap.parse_args()
    a=json.load(open(os.path.join(args.run,'args.json')))
    dev='cuda'
    m=timm.create_model(a['model'],pretrained=False,num_classes=5,drop_path_rate=a.get('drop_path',0.0)).cuda()
    m.load_state_dict(torch.load(os.path.join(args.run,args.ckpt),map_location='cpu')['model']); m.eval()
    d=pd.read_csv(a['split']); va=d[d.fold==a['fold']].reset_index(drop=True)
    combos={
      'id':((0,),(1.0,)),
      'h':((0,1),(1.0,)),
      'hv':((0,1,2,3),(1.0,)),
    }
    if not args.flips_only:
        combos['h_s115']=((0,1),(1.0,1.15))
        combos['hv_s115']=((0,1,2,3),(1.0,1.15))
    res={}
    for name,(fl,sc) in combos.items():
        p=infer(m,va,os.path.join(DATA,'train_images'),a['size'],args.bs,args.workers,fl,sc)
        acc=float((p.argmax(1)==va.label.values).mean()); res[name]=acc; print(name,round(acc,5),flush=True)
    best=max(res,key=res.get); print('BEST',best,res[best],flush=True)
    if args.test:
        od=args.outdir or os.path.join(args.run,'tta_'+best); os.makedirs(od,exist_ok=True)
        fl,sc=combos[best]
        vp=infer(m,va,os.path.join(DATA,'train_images'),a['size'],args.bs,args.workers,fl,sc)
        np.save(os.path.join(od,'val_probs_tta.npy'),vp); open(os.path.join(od,'val_acc.txt'),'w').write(str(res[best]))
        te=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
        tp=infer(m,te,os.path.join(DATA,'test_images'),a['size'],args.bs,args.workers,fl,sc)
        np.save(os.path.join(od,'test_probs.npy'),tp); te.assign(label=tp.argmax(1)).to_csv(os.path.join(od,'submission.csv'),index=False)
        print('saved',od,flush=True)

if __name__=='__main__': main()
