import os, sys
import numpy as np, pandas as pd

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')
DATA=os.path.join(HERE,'..','..','data')

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def parse(cid):
    parts=cid.str.split('_')
    return (parts.str[0]+'_'+parts.str[1]).values, parts.str[2].astype(int).values, parts.str[3].values, parts.str[4].values

def smooth_arr(cid, prob, w):
    gp,step,p1,p2=parse(pd.Series(cid))
    df=pd.DataFrame({'contact_id':cid,'game_play':gp,'step':step,'p1':p1,'p2':p2,'prob':prob})
    df=df.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    r=df.groupby(['game_play','p1','p2'],sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    df['sm']=r.reset_index(level=[0,1,2],drop=True).reindex(df.index).values
    return df

def main(tag='l2', W=5):
    ids=np.load(os.path.join(CACHE,f'ids_{tag}.npy'),allow_pickle=True)
    oof=np.load(os.path.join(CACHE,f'oof_{tag}.npy')).astype(np.float64)
    pids=np.load(os.path.join(CACHE,f'pids_{tag}.npy'),allow_pickle=True)
    pt=np.load(os.path.join(CACHE,f'ptest_{tag}.npy')).astype(np.float64)
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    m=pd.DataFrame({'contact_id':ids,'prob':oof})
    dfo=smooth_arr(m.contact_id.values,m.prob.values,W).merge(lab,on='contact_id')
    mcc,thr=best_mcc(dfo.contact.values,dfo.sm.values)
    print('%s OOF smooth w%d MCC %.5f thr %.5f'%(tag,W,mcc,thr),flush=True)
    tdf=smooth_arr(pids,pt,W)
    tdf['pred']=(tdf.sm.values>=thr).astype(int)
    ss=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    out=tdf.set_index('contact_id')['pred'].reindex(ss.contact_id).reset_index()
    out.columns=['contact_id','contact']
    assert out.contact.notna().all() and len(out)==len(ss)
    out.to_csv(os.path.join(HERE,f'sub_{tag}.csv'),index=False)
    print('test pos rate %.4f rows %d'%(out.contact.mean(),len(out)),flush=True)
    out.to_csv(os.path.join(HERE,'..','..','final_solution','submission.csv'),index=False)
    print('wrote final_solution/submission.csv',flush=True)
    pd.DataFrame({'contact_id':ids,'prob':oof}).to_parquet(os.path.join(HERE,f'oof_pred_{tag}.parquet'))
    pd.DataFrame({'contact_id':pids,'prob':pt}).to_parquet(os.path.join(HERE,f'test_pred_{tag}.parquet'))
    print('artifacts written',flush=True)

if __name__=='__main__':
    main()
