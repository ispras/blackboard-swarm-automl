import os, sys, json
import numpy as np, pandas as pd

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def smooth_vals(tmp, groups, w):
    r=tmp.groupby(list(groups),sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    return r.reset_index(level=[0,1,2],drop=True).reindex(tmp.index).values

def evaluate(tag, mask_ground=None):
    ids=np.load(os.path.join(CACHE,f'ids_{tag}.npy'),allow_pickle=True); oof=np.load(os.path.join(CACHE,f'oof_{tag}.npy'))
    if len(ids)!=len(oof): print('len mismatch',tag); return
    # load contact from featg? use full ids->contact map from train_labels
    lab=pd.read_csv(os.path.join(HERE,'..','..','data','train_labels.csv'), usecols=['contact_id','contact'])
    m=pd.DataFrame({'contact_id':ids,'prob':oof}).merge(lab,on='contact_id',how='left')
    y=m.contact.values
    print(f'=== {tag}: rows {len(m)} pos {y.mean():.4f}')
    mm,tt=best_mcc(y,oof); print(f'  raw MCC {mm:.5f} thr {tt:.4f}')
    tmp=m[['contact_id','prob','contact']].copy()
    parts=tmp.contact_id.str.split('_')
    tmp['game_play']=parts.str[0]+'_'+parts.str[1]; tmp['p1']=parts.str[3]; tmp['p2']=parts.str[4]; tmp['step']=parts.str[2].astype(int)
    tmp=tmp.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    best=(-1,0,0)
    for w in (1,3,5,7,9,11):
        pp=smooth_vals(tmp,('game_play','p1','p2'),w); ms,ts=best_mcc(tmp.contact.values,pp)
        print(f'  smooth w{w}: {ms:.5f} thr {ts:.4f}')
        if ms>best[0]: best=(ms,w,ts)
    print(f'  BEST smooth {best[0]:.5f} w{best[1]} thr {best[2]:.4f}')
    return best

if __name__=='__main__':
    for tag in sys.argv[1:]:
        try: evaluate(tag)
        except Exception as e: print(tag,'ERR',e)
