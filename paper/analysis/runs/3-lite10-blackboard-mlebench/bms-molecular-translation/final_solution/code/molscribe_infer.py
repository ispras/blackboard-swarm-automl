import sys, os, time, argparse
sys.path.insert(0, '/download_job_code/payload/sandbox/workspaces/altair_epsilon/pylibs')
sys.path.insert(0, '/usr/local/lib/python3.11/site-packages')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MolScribe'))
import torch, pandas as pd, numpy as np, cv2
from molscribe import MolScribe
ap=argparse.ArgumentParser()
ap.add_argument('--ids', required=True)
ap.add_argument('--imgroot', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--ckpt', default='weights/swin_base_char_aux_1m.pth')
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshard', type=int, default=1)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--conf', action='store_true')
args=ap.parse_args()
def path(root,i): return f"{root}/{i[0]}/{i[1]}/{i[2]}/{i}.png"
model=MolScribe(args.ckpt, device=torch.device('cuda'), num_workers=1)
if args.conf: model.decoder.compute_confidence=True
df=pd.read_csv(args.ids)
df=df.iloc[args.shard::args.nshard].reset_index(drop=True)
if args.limit: df=df.iloc[:args.limit]
ids=df.image_id.tolist()
if os.path.exists(args.out):
    done=set(pd.read_csv(args.out).image_id.astype(str))
    ids=[i for i in ids if i not in done]
    print('resume: skipping',len(done),'remaining',len(ids),flush=True)
rows=[]; t0=time.time()
def flush():
    pd.DataFrame(rows,columns=['image_id','smiles','score']).to_csv(args.out,index=False)
for k in range(0,len(ids),args.bs):
    chunk=ids[k:k+args.bs]; imgs=[]; ok=[]
    for i in chunk:
        im=cv2.imread(path(args.imgroot,i))
        if im is None: continue
        imgs.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); ok.append(i)
    if not imgs: continue
    ims=[model.transform(image=im,keypoints=[])['image'] for im in imgs]
    x=torch.stack(ims).cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        feats,hid=model.encoder(x)
        preds=model.decoder.decode(feats,hid)
    for i,p in zip(ok,preds):
        rows.append((i,p['chartok_coords']['smiles'], float(p.get('overall_score',0.0))))
    done=k+len(chunk)
    if done % 2000 < args.bs:
        flush()
        el=time.time()-t0; print(f'{done}/{len(ids)} {done/el:.2f} img/s', flush=True)
flush()
print('DONE',len(rows),'in',round(time.time()-t0,1),'s', round(len(rows)/(time.time()-t0),2),'img/s', flush=True)
