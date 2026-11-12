import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import matthews_corrcoef
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache'); DATA=os.path.join(HERE,'..','..','data'); WS=os.path.join(HERE,'..')

def keys_of(cid):
    p=cid.str.split('_'); return (p.str[0]+'_'+p.str[1]), p.str[2].astype(int), p.str[3], p.str[4]
def discover():
    out={}
    for p in glob.glob(os.path.join(WS,'*','oof_pred*.parquet')):
        n=os.path.basename(os.path.dirname(p))+'_'+os.path.basename(p).replace('oof_pred','').replace('.parquet','').strip('_')
        if os.path.exists(p.replace('oof_pred','test_pred')): out[n]=p
    return out
def smooth_by(df,w,keys):
    df=df.sort_values(list(keys)+['step']).reset_index(drop=True)
    r=df.groupby(list(keys),sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    df['sm']=r.reset_index(level=list(range(len(keys))),drop=True).reindex(df.index).values
    return df
def main():
    cfg=json.load(open(os.path.join(CACHE,'final_opt.json')))
    wg,wp,qg,qp,wgr,wpr=cfg['wg'],cfg['wp'],cfg['qg'],cfg['qp'],cfg['wgr'],cfg['wpr']
    arts=discover(); names=set(wg)|set(wp)
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    ref=pd.read_parquet(os.path.join(HERE,'oof_pred.parquet'),columns=['contact_id']).merge(lab,on='contact_id',how='left')
    ref['game_play'],ref['step'],ref['p1'],ref['p2']=keys_of(ref.contact_id)
    ref=ref.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    R={}
    for n in names:
        o=pd.read_parquet(arts[n]).drop_duplicates('contact_id').set_index('contact_id')
        v=np.nan_to_num(o['prob'].reindex(ref.contact_id.values).values,nan=0.0); R[n]=rankdata(v)/len(v)
    def blend(w):
        s=np.zeros(len(ref))
        for n,c in w.items(): s+=R[n]*c
        return s/sum(w.values())
    isg=(ref.p2=='G').values; y=ref.contact.values.astype(int)
    dg=ref[isg][['game_play','step','p1','p2','contact']].copy(); dg['prob']=blend(wg)[isg]; dg=smooth_by(dg,wgr,('game_play','p1'))
    dp=ref[~isg][['game_play','step','p1','p2','contact']].copy(); dp['prob']=blend(wp)[~isg]; dp=smooth_by(dp,wpr,('game_play','p1','p2'))
    tg=np.quantile(dg.sm.values,1-qg); tp=np.quantile(dp.sm.values,1-qp)
    pg=(dg.sm.values>=tg).astype(int); pp=(dp.sm.values>=tp).astype(int)
    pred=np.empty(len(ref),int); pred[isg]=pg; pred[~isg]=pp
    print('sklearn OOF MCC %.6f'%matthews_corrcoef(y,pred))
    print('ground MCC %.5f | pair MCC %.5f'%(matthews_corrcoef(y[isg],pg),matthews_corrcoef(y[~isg],pp)))
    print('qg %.4f qp %.4f'%(pg.mean(),pp.mean()))
if __name__=='__main__': main()
