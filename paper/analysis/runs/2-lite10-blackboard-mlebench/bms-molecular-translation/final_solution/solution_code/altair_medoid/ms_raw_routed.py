import os,sys,time,argparse,cv2,numpy as np,pandas as pd,torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from predict import build_model
from infer_iota import to_inchi_parallel
ROOT='/download_job_code/payload/sandbox'
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ids',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--bs',type=int,default=128); ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args()
    ids=pd.read_csv(a.ids).image_id.astype(str).tolist()
    if a.limit: ids=ids[:a.limit]
    paths=[f'{ROOT}/data/test/{i[0]}/{i[1]}/{i[2]}/{i}.png' for i in ids]
    raw=a.out+'.raw.csv'
    done=set()
    if os.path.exists(raw):
        try: done=set(pd.read_csv(raw,usecols=['image_id']).image_id.astype(str)); print('resume',len(done),flush=True)
        except Exception: pass
    order=[j for j in np.argsort([os.path.getsize(p) if os.path.exists(p) else 0 for p in paths]) if ids[j] not in done]
    model=build_model(torch.device('cuda'))
    buf=[]; t0=time.time(); n=0
    def flush():
        nonlocal buf
        if not buf: return
        pd.DataFrame(buf,columns=['image_id','smiles','conf']).to_csv(raw,mode='a',header=not (os.path.exists(raw) and os.path.getsize(raw)>0),index=False)
        buf=[]
    with torch.inference_mode():
        for bi in range(0,len(order),a.bs):
            js=order[bi:bi+a.bs]; imgs=[]
            for j in js:
                im=cv2.imread(paths[j])
                imgs.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB) if im is not None else None)
            keep=[k for k,im in enumerate(imgs) if im is not None]; b=[imgs[k] for k in keep]
            if not b: continue
            try:
                x=torch.stack([model.transform(image=z,keypoints=[])['image'] for z in b]).cuda()
                f,h=model.encoder(x); preds=model.decoder.decode(f,h)
            except Exception as e:
                print('batch err',e,flush=True); continue
            for k,p in zip(keep,preds):
                buf.append((ids[js[k]], p['chartok_coords']['smiles'], float(p.get('overall_score',-1.0))))
            n+=len(b)
            if (bi//a.bs)%5==0:
                flush(); el=time.time()-t0; print(f'{n}/{len(order)} {n/el:.1f} img/s eta {(len(order)-n)/max(n/el,1e-6)/60:.1f}m',flush=True)
    flush()
    print('decoded',len(order),'time',round(time.time()-t0,1),flush=True)
    d=pd.read_csv(raw).drop_duplicates('image_id')
    inc=to_inchi_parallel(d.smiles.fillna('').astype(str).tolist(),workers=32)
    out=pd.DataFrame({'image_id':d.image_id,'smiles':d.smiles,'conf':d.conf,'inchi':inc})
    out['valid']=out.inchi.astype(str).str.len()>0
    out.to_csv(a.out,index=False)
    print('saved',a.out,out.shape,'valid',out.valid.mean(),flush=True)
if __name__=='__main__': main()
