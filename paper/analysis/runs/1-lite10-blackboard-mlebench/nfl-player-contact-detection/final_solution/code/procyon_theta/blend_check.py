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

def load(path, pcol='prob'):
    df=pd.read_parquet(path)
    parts=df.contact_id.str.split('_')
    df['game_play']=parts.str[0]+'_'+parts.str[1]; df['step']=parts.str[2].astype(int)
    df['p1']=parts.str[3]; df['p2']=parts.str[4]
    df=df.rename(columns={pcol:'prob'})
    return df[['contact_id','game_play','step','p1','p2','prob']]

def main():
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    base=load(os.path.join(HERE,'oof_pred.parquet'))
    l2=np.load(os.path.join(CACHE,'ids_l2.npy'),allow_pickle=True); l2p=np.load(os.path.join(CACHE,'oof_l2.npy'))
    l2d=pd.DataFrame({'contact_id':l2,'l2':l2p})
    seq=np.load(os.path.join(CACHE,'ids_seq.npy'),allow_pickle=True); seqp=np.load(os.path.join(CACHE,'oof_seq.npy'))
    seqd=pd.DataFrame({'contact_id':seq,'seq':seqp})
    df=base.merge(l2d,on='contact_id').merge(seqd,on='contact_id').merge(lab,on='contact_id')
    print('merged',len(df),'pos',df.contact.mean(),flush=True)
    r_base=rankdata(df.prob.values)/len(df); r_l2=rankdata(df.l2.values)/len(df); r_seq=rankdata(df.seq.values)/len(df)
    isg=(df.p2=='G').values
    def ev(name,score):
        t=df[['game_play','p1','p2','step','contact']].copy(); t['prob']=score
        out=[]
        for w in (5,7,9):
            s=smooth(t,w); m,th=best_mcc(s.contact.values,s.sm.values); out.append((m,w,th))
        m,w,th=max(out)
        print('%-28s smooth MCC %.5f w%d thr %.4f'%(name,m,w,th),flush=True)
    ev('L2',df.l2.values)
    ev('rank L2',r_l2)
    ev('rank seq',r_seq)
    for a in [0.1,0.2,0.3]:
        ev('L2 + %.1f seq'%a, (1-a)*r_l2+a*r_seq)
        ev('base + %.1f seq'%a, (1-a)*r_base+a*r_seq)
    for a in [0.1,0.2,0.3]:
        ev('L2 + %.1f seq(ground only)'%a, np.where(isg,(1-a)*r_l2+a*r_seq,r_l2))

if __name__=='__main__':
    main()
