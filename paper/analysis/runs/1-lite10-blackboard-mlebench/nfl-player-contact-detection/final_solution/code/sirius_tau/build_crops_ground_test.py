import pandas as pd, numpy as np, cv2, os, time
from multiprocessing import Pool

OUT='workspaces/sirius_tau'
HEL='workspaces/fomalhaut_omicron/cache/hel_test.parquet'
VDIR='data/test'
FPS=59.94
T=7; HH=WW=64
VIEWS=['Endzone','Sideline']
CPATH=f'{OUT}/crops/crops_ground_test.npy'

def build_index():
    ss=pd.read_csv('data/sample_submission.csv')
    p=ss.contact_id.str.rsplit('_',n=3,expand=True)
    idx=pd.DataFrame({'game_play':p[0],'step':p[1].astype(int),'nfl_player_id_1':p[2]})
    # ground rows have p2=='G'; parse again
    idx['p2']=p[3].values
    idx=idx[idx.p2=='G'].drop(columns='p2').reset_index(drop=True)
    idx=idx.sort_values(['game_play','step','nfl_player_id_1']).reset_index(drop=True)
    idx['row']=np.arange(len(idx))
    return idx

HEL_D=None; META=None; MM=None; hd_steps={}
def init():
    global HEL_D, META, MM
    md=pd.read_csv('data/test_video_metadata.csv'); META={}
    for _,r in md.iterrows():
        st=pd.Timestamp(r['start_time']); sn=pd.Timestamp(r['snap_time'])
        META[(r['game_play'],r['view'])]=(sn-st).total_seconds()*FPS
    h=pd.read_parquet(HEL); HEL_D={}
    for (gp,v),sub in h.groupby(['game_play','view'],sort=False):
        d={}
        for s,ss in sub.groupby('step',sort=False):
            d[int(s)]={int(r.nfl_player_id):(r.cx,r.cy,r.top,r.bottom,r.width,r.height) for r in ss.itertuples()}
        HEL_D[(gp,v)]=d
    MM=np.load(CPATH,mmap_mode='r+')

def worker(args):
    gp, rows = args
    out=np.zeros((len(rows),T,2,HH,WW,3),dtype=np.uint8); fpv={0:{},1:{}}
    for li,(row,step,p1) in enumerate(rows):
        step=int(step); p1=int(p1)
        for vi,vname in enumerate(VIEWS):
            base=META.get((gp,vname))
            if base is None: continue
            hd=HEL_D.get((gp,vname))
            if not hd: continue
            steps=hd_steps.get((gp,vname))
            if steps is None:
                steps=sorted(hd.keys()); hd_steps[(gp,vname)]=steps
            for o in range(T):
                t=step+o-(T//2)
                if t in hd: ts=t
                else:
                    j=np.searchsorted(steps,t); cand=[]
                    if j<len(steps): cand.append(steps[j])
                    if j>0: cand.append(steps[j-1])
                    if not cand: continue
                    ts=min(cand,key=lambda x:abs(x-t))
                    if abs(ts-t)>3: continue
                b=hd[ts].get(p1)
                if b is None: continue
                cx,cy,tp,bt,w,hh=b
                L,R,Tp,B=cx-w/2,cx+w/2,tp,bt
                pad=1.6*max(w,hh)+12
                L-=pad; R+=pad; Tp-=pad; B+=pad
                fr=int(round(base+t*0.1*FPS))
                fpv[vi].setdefault(fr,[]).append((li,o,(L,Tp,R,B)))
    for vi,vname in enumerate(VIEWS):
        need=fpv[vi]
        if not need: continue
        vf=f'{VDIR}/{gp}_{vname}.mp4'
        if not os.path.exists(vf): continue
        cap=cv2.VideoCapture(vf); fi=0; maxf=max(need.keys())
        while fi<=maxf:
            ok,img=cap.read()
            if not ok: break
            if fi in need:
                H,W=img.shape[:2]
                for (li,o,(L,Tp,R,B)) in need[fi]:
                    Li=int(max(0,np.floor(L))); Ri=int(min(W,np.ceil(R)))
                    Ti=int(max(0,np.floor(Tp))); Bi=int(min(H,np.ceil(B)))
                    if Ri-Li<4 or Bi-Ti<4: continue
                    out[li,o,vi]=cv2.resize(img[Ti:Bi,Li:Ri],(WW,HH),interpolation=cv2.INTER_AREA)
            fi+=1
        cap.release()
    base_row=rows[0][0]; MM[base_row:base_row+len(rows)]=out
    return gp

def main():
    idx=build_index()
    print('ground test samples',len(idx),flush=True)
    idx.to_parquet(f'{OUT}/ground_test_index.parquet')
    N=len(idx); os.makedirs(f'{OUT}/crops',exist_ok=True)
    mm=np.lib.format.open_memmap(CPATH,mode='w+',dtype=np.uint8,shape=(N,T,2,HH,WW,3)); del mm
    print('allocated GB',N*T*2*HH*WW*3/1e9,flush=True)
    jobs=[(p,sub[['row','step','nfl_player_id_1']].values.tolist()) for p,sub in idx.groupby('game_play',sort=False)]
    t0=time.time()
    with Pool(8,initializer=init) as pool:
        for k,_ in enumerate(pool.imap_unordered(worker,jobs)):
            print('plays',k,'t',round(time.time()-t0),flush=True)
    print('DONE',round(time.time()-t0),flush=True)
    mm2=np.load(CPATH,mmap_mode='r'); smp=np.asarray(mm2[::777])
    print('verify nonzero',float((smp>0).mean()),flush=True)

if __name__=='__main__': main()
