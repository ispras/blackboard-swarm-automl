import numpy as np, pandas as pd, json, sys
from sklearn.metrics import matthews_corrcoef
W='workspaces/capella_pi'
OOF=sys.argv[1] if len(sys.argv)>1 else f'{W}/oof_stack_v2.parquet'
TEST=sys.argv[2] if len(sys.argv)>2 else f'{W}/test_stack_v2.parquet'
OUTP=sys.argv[3] if len(sys.argv)>3 else f'{W}/submission_stack_v2.csv'
OCOL='stack_oof'; TCOL='stack_pred'

def best_mcc(y,p):
    ths=np.quantile(p,np.linspace(0.5,0.9995,300)); order=np.argsort(-p); ps=p[order]; ys=np.asarray(y)[order]
    cp=np.cumsum(ys,dtype=np.float64); cn=np.cumsum(1-ys,dtype=np.float64)
    Pos=int(ys.sum()); Neg=len(ys)-Pos
    idx=np.clip(np.searchsorted(-ps,-ths,side='left'),0,len(ys))
    tp=np.where(idx>0,cp[np.maximum(idx-1,0)],0.0); fp=np.where(idx>0,cn[np.maximum(idx-1,0)],0.0)
    fn=Pos-tp; tn=Neg-fp; den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+1e-12
    mcc=(tp*tn-fp*fn)/den; k=int(np.argmax(mcc)); return float(mcc[k]),float(ths[k])

def keys(df):
    sp=df.contact_id.str.split('_')
    return pd.DataFrame({'gp':(sp.str[0]+'_'+sp.str[1]).values,'step':sp.str[2].astype(int).values,
                         'p1':sp.str[3].values,'p2':sp.str[4].values},index=df.index)

def smooth(df,pcol,w):
    if w<=0: return df[pcol].values
    d=pd.DataFrame({'gp':df.gp.values,'p1':df.p1.values,'p2':df.p2.values,'s':df.step.values,'p':df[pcol].values})
    d=d.sort_values(['gp','p1','p2','s']); g=d.groupby(['gp','p1','p2'],sort=False)['p']
    r=g.rolling(w,center=True,min_periods=1).mean(); r.index=r.index.droplevel([0,1,2])
    return r.reindex(d.index).sort_index().values

oof=pd.read_parquet(OOF); test=pd.read_parquet(TEST)
for d in (oof,test):
    k=keys(d)
    for c in k: d[c]=k[c].values
    if 'is_ground' not in d: d['is_ground']=(d.p2=='G').astype(int)
res={}
for task,igmask in [('pp',0),('ground',1)]:
    m=oof.is_ground.values==igmask
    pbest=(-1,0.5,0)
    for w in [0,1,3,5,7]:
        p=smooth(oof[m],OCOL,w); mcc,th=best_mcc(oof.contact.values[m],p)
        if mcc>pbest[0]: pbest=(mcc,th,w)
    res[task]=pbest; print(task,'mcc',round(pbest[0],5),'th',round(pbest[1],5),'w',pbest[2])
# combined
ig=oof.is_ground.values==1
mpp,thpp,wpp=res['pp']; mg,thg,wg=res['ground']
ppp=smooth(oof[~ig],OCOL,wpp); pg=smooth(oof[ig],OCOL,wg)
yg=np.concatenate([oof.contact.values[~ig],oof.contact.values[ig]])
pred=np.concatenate([ppp>thpp,pg>thg]); glob=matthews_corrcoef(yg,pred)
print('GLOBAL',round(glob,5))
# test
te_pred=np.zeros(len(test),dtype=np.int8)
for task,igmask,th,w in [('pp',0,thpp,wpp),('ground',1,thg,wg)]:
    m=test.is_ground.values==igmask
    p=smooth(test[m],TCOL,w); te_pred[m]=(p>th).astype(np.int8)
    print('test',task,'n',int(m.sum()),'pos',int(te_pred[m].sum()),'rate',round(float(te_pred[m].mean()),5))
out=pd.DataFrame({'contact_id':test.contact_id.values,'contact':te_pred})
ss=pd.read_csv('data/sample_submission.csv')[['contact_id']]
out=ss.merge(out,on='contact_id',how='left')
assert out.contact.notna().all(), out.contact.isna().sum()
out['contact']=out.contact.astype(np.int8)
out.to_csv(OUTP,index=False)
print('wrote',OUTP,len(out),'global pos rate',round(float(out.contact.mean()),5))
json.dump({'pp':res['pp'],'ground':res['ground'],'global':glob},open(f'{W}/submission_meta.json','w'),indent=1,default=float)
