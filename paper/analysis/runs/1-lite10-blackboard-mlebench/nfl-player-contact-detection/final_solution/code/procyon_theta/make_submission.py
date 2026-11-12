import os, sys, json
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

def parse(df):
    parts=df.contact_id.str.split('_')
    df=df.copy()
    df['game_play']=parts.str[0]+'_'+parts.str[1]; df['step']=parts.str[2].astype(int)
    df['p1']=parts.str[3]; df['p2']=parts.str[4]
    return df

def smooth(df,w):
    df=df.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    r=df.groupby(['game_play','p1','p2'],sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    df['sm']=r.reset_index(level=[0,1,2],drop=True).reindex(df.index).values
    return df

def main():
    W=7
    oof=parse(pd.read_parquet(os.path.join(HERE,'oof_pred.parquet')))
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    oof=oof.merge(lab,on='contact_id',how='left')
    s=smooth(oof[['contact_id','game_play','p1','p2','step','prob','contact']].copy(),W)
    mcc,thr=best_mcc(s.contact.values,s.sm.values)
    print('OOF gall smooth w%d MCC %.5f thr %.5f'%(W,mcc,thr),flush=True)
    tst=parse(pd.read_parquet(os.path.join(HERE,'test_pred.parquet')))
    st=smooth(tst[['contact_id','game_play','p1','p2','step','prob']].copy(),W)
    st['pred']=(st.sm.values>=thr).astype(int)
    ss=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    out=st.set_index('contact_id')['pred'].reindex(ss.contact_id).reset_index()
    out.columns=['contact_id','contact']
    assert len(out)==len(ss) and out.contact.notna().all()
    out.to_csv(os.path.join(HERE,'sub_gall.csv'),index=False)
    print('test pos rate',out.contact.mean(),'rows',len(out),flush=True)
    out.to_csv(os.path.join(HERE,'..','..','final_solution','submission.csv'),index=False)
    print('wrote final_solution/submission.csv',flush=True)

if __name__=='__main__':
    main()
