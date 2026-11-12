import pandas as pd, numpy as np, cv2, os, sys, json, time
from multiprocessing import Pool

OUT='workspaces/sirius_tau'
HEL='workspaces/fomalhaut_omicron/cache/hel_train.parquet'
FEAT='workspaces/fomalhaut_omicron/cache/feat_train.parquet'
VDIR='data/train'
FPS=59.94
T=5
HH=WW=64
VIEWS=['Endzone','Sideline']
NEG_DIST_MAX=2.2
SEED=0
CPATH=f'{OUT}/crops/crops_train.npy'

def build_index():
    idx = pd.read_parquet(FEAT, columns=['game_play','step','nfl_player_id_1','nfl_player_id_2','dist','contact'])
    idx = idx[idx.nfl_player_id_2.astype(str)!='G'].copy()
    idx['contact']=idx.contact.astype(np.int8)
    pos = idx[idx.contact==1]
    neg = idx[(idx.contact==0)&(idx.dist<NEG_DIST_MAX)]
    neg = neg.sample(n=min(len(neg),int(os.environ.get('NEG','120000'))), random_state=SEED)
    idx = pd.concat([pos,neg]).sort_values(['game_play','step','nfl_player_id_1','nfl_player_id_2']).reset_index(drop=True)
    idx['row']=np.arange(len(idx))
    return idx

HEL_D=None; META=None; MM=None

def init():
    global HEL_D, META, MM
    md=pd.read_csv('data/train_video_metadata.csv')
    META={}
    for _,r in md.iterrows():
        st=pd.Timestamp(r['start_time']); sn=pd.Timestamp(r['snap_time'])
        META[(r['game_play'],r['view'])]=(sn-st).total_seconds()*FPS
    h=pd.read_parquet(HEL)
    HEL_D={}
    for (gp,v),sub in h.groupby(['game_play','view'],sort=False):
        d={}
        for s,ss in sub.groupby('step',sort=False):
            d[int(s)]={int(r.nfl_player_id):(r.cx,r.cy,r.top,r.bottom,r.width,r.height) for r in ss.itertuples()}
        HEL_D[(gp,v)]=d
    MM=np.load(CPATH, mmap_mode='r+')

def worker(args):
    gp, rows = args
    out=np.zeros((len(rows),T,2,HH,WW,3),dtype=np.uint8)
    frames_per_view={0:{},1:{}}
    for li,(row,step,p1,p2) in enumerate(rows):
        step=int(step); p1=int(p1); p2=int(p2)
        for vi,vname in enumerate(VIEWS):
            base=META.get((gp,vname))
            if base is None: continue
            hd=HEL_D.get((gp,vname))
            if not hd: continue
            steps=hd_steps.setdefault((gp,vname),None)
            if steps is None:
                steps=sorted(hd.keys()); hd_steps[(gp,vname)]=steps
            for o in range(T):
                t=step+o-(T//2)
                if t in hd: ts=t
                else:
                    j=np.searchsorted(steps,t)
                    cand=[]
                    if j<len(steps): cand.append(steps[j])
                    if j>0: cand.append(steps[j-1])
                    if not cand: continue
                    ts=min(cand,key=lambda x:abs(x-t))
                    if abs(ts-t)>3: continue
                b=hd[ts]
                b1=b.get(p1); b2=b.get(p2)
                if b1 is None and b2 is None: continue
                if b1 is None: b1=b2
                if b2 is None: b2=b1
                _,cx1,tp1,bt1,w1,h1=b1; _,cx2,tp2,bt2,w2,h2=b2
                L=min(cx1-w1/2,cx2-w2/2); R=max(cx1+w1/2,cx2+w2/2)
                Tp=min(tp1,tp2); B=max(bt1,bt2)
                pad=0.6*max(R-L,B-Tp)+8
                L-=pad; R+=pad; Tp-=pad; B+=pad
                fr=int(round(base+t*0.1*FPS))
                frames_per_view[vi].setdefault(fr,[]).append((li,o,(L,Tp,R,B)))
    for vi,vname in enumerate(VIEWS):
        need=frames_per_view[vi]
        if not need: continue
        vf=f'{VDIR}/{gp}_{vname}.mp4'
        if not os.path.exists(vf): continue
        cap=cv2.VideoCapture(vf)
        fi=0; maxf=max(need.keys())
        while fi<=maxf:
            ok,img=cap.read()
            if not ok: break
            if fi in need:
                H,W=img.shape[:2]
                for (li,o,(L,Tp,R,B)) in need[fi]:
                    Li=int(max(0,np.floor(L))); Ri=int(min(W,np.ceil(R)))
                    Ti=int(max(0,np.floor(Tp))); Bi=int(min(H,np.ceil(B)))
                    if Ri-Li<4 or Bi-Ti<4: continue
                    crop=img[Ti:Bi,Li:Ri]
                    out[li,o,vi]=cv2.resize(crop,(WW,HH),interpolation=cv2.INTER_AREA)
            fi+=1
        cap.release()
    base_row=rows[0][0]
    MM[base_row:base_row+len(rows)]=out
    return gp

hd_steps={}

def main():
    idx=build_index()
    print('samples',len(idx),'pos',int(idx.contact.sum()),flush=True)
    idx.to_parquet(f'{OUT}/crops_index.parquet')
    N=len(idx)
    os.makedirs(f'{OUT}/crops',exist_ok=True)
    mm=np.lib.format.open_memmap(CPATH,mode='w+',dtype=np.uint8,shape=(N,T,2,HH,WW,3)); del mm
    print('allocated GB',N*T*2*HH*WW*3/1e9,flush=True)
    jobs=[]
    for p,sub in idx.groupby('game_play',sort=False):
        jobs.append((p, sub[['row','step','nfl_player_id_1','nfl_player_id_2']].values.tolist()))
    t0=time.time()
    with Pool(8,initializer=init) as pool:
        for k,_ in enumerate(pool.imap_unordered(worker,jobs)):
            if k%25==0: print('plays',k,'t',round(time.time()-t0,1),flush=True)
    print('DONE',round(time.time()-t0,1),flush=True)
    mm2=np.load(CPATH,mmap_mode='r')
    smp=np.asarray(mm2[::777])
    print('verify nonzero frac',float((smp>0).mean()),'mean',float(smp.mean()),flush=True)

if __name__=='__main__':
    main()
