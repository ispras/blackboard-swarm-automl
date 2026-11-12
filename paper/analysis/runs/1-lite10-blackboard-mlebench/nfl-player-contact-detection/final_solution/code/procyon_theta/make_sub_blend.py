import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache'); DATA=os.path.join(HERE,'..','..','data')

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def smooth(df,w):
    df=df.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    r=df.groupby(['game_play','p1','p2'],sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    df['sm']=r.reset_index(level=[0,1,2],drop=True).reindex(df.index).values
    return df

def keys(cid):
    parts=cid.str.split('_')
    return (parts.str[0]+'_'+parts.str[1]), parts.str[2].astype(int), parts.str[3], parts.str[4]

def main():
    W=5
    l2i=np.load(os.path.join(CACHE,'ids_l2.npy'),allow_pickle=True); l2p=np.load(os.path.join(CACHE,'oof_l2.npy'))
    sqi=np.load(os.path.join(CACHE,'ids_seq.npy'),allow_pickle=True); sqp=np.load(os.path.join(CACHE,'oof_seq.npy'))
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    df=pd.DataFrame({'contact_id':l2i,'l2':l2p}).merge(pd.DataFrame({'contact_id':sqi,'seq':sqp}),on='contact_id').merge(lab,on='contact_id')
    df['game_play'],df['step'],df['p1'],df['p2']=keys(df.contact_id)
    r_l2=rankdata(df.l2.values)/len(df); r_seq=rankdata(df.seq.values)/len(df)
    best=(-1,0,0,0)
    for a in [0.0,0.1,0.15,0.2,0.25,0.3]:
        for w in (5,7,9):
            sc=(1-a)*r_l2+a*r_seq
            t=df[['game_play','p1','p2','step','contact']].copy(); t['prob']=sc
            s=smooth(t,w); m,th=best_mcc(s.contact.values,s.sm.values)
            if m>best[0]: best=(m,a,w,th)
    m,a,w,th=best
    print('OOF best blend alpha=%.2f w%d MCC %.5f thr %.5f'%(a,w,m,th),flush=True)
    # test
    t2i=np.load(os.path.join(CACHE,'pids_l2.npy'),allow_pickle=True); t2p=np.load(os.path.join(CACHE,'ptest_l2.npy'))
    tsi=np.load(os.path.join(CACHE,'pids_seq.npy'),allow_pickle=True); tsp=np.load(os.path.join(CACHE,'ptest_seq.npy'))
    td=pd.DataFrame({'contact_id':t2i,'l2':t2p}).merge(pd.DataFrame({'contact_id':tsi,'seq':tsp}),on='contact_id')
    td['game_play'],td['step'],td['p1'],td['p2']=keys(td.contact_id)
    rt_l2=rankdata(td.l2.values)/len(td); rt_seq=rankdata(td.seq.values)/len(td)
    td['prob']=(1-a)*rt_l2+a*rt_seq
    st=smooth(td[['contact_id','game_play','p1','p2','step','prob']].copy(),w)
    st['pred']=(st.sm.values>=th).astype(int)
    ss=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    out=st.set_index('contact_id')['pred'].reindex(ss.contact_id).reset_index(); out.columns=['contact_id','contact']
    assert out.contact.notna().all() and len(out)==len(ss)
    out.to_csv(os.path.join(HERE,'sub_l2_seq.csv'),index=False)
    print('test pos rate %.4f'%out.contact.mean(),flush=True)
    out.to_csv(os.path.join(HERE,'..','..','final_solution','submission.csv'),index=False)
    print('wrote final_solution/submission.csv',flush=True)
    # publish seq artifacts
    pd.DataFrame({'contact_id':sqi,'prob':sqp}).to_parquet(os.path.join(HERE,'oof_pred_seq.parquet'))
    pd.DataFrame({'contact_id':tsi,'prob':tsp}).to_parquet(os.path.join(HERE,'test_pred_seq.parquet'))
    print('seq artifacts written',flush=True)

if __name__=='__main__':
    main()
