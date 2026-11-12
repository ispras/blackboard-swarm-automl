import pandas as pd, numpy as np, json, os, sys
from sklearn.metrics import matthews_corrcoef
def logit(p,e=1e-6): p=np.clip(np.asarray(p,dtype=float),e,1-e); return np.log(p/(1-p))
def best_mcc(y,p):
    ths=np.quantile(p,np.linspace(0.5,0.9995,300)); order=np.argsort(-p); ps=p[order]; ys=np.asarray(y)[order]
    cp=np.cumsum(ys,dtype=np.float64); cn=np.cumsum(1-ys,dtype=np.float64)
    Pos=int(ys.sum()); Neg=len(ys)-Pos
    idx=np.clip(np.searchsorted(-ps,-ths,side='left'),0,len(ys))
    tp=np.where(idx>0,cp[np.maximum(idx-1,0)],0.0); fp=np.where(idx>0,cn[np.maximum(idx-1,0)],0.0)
    fn=Pos-tp; tn=Neg-fp; den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+1e-12
    mcc=(tp*tn-fp*fn)/den; k=int(np.argmax(mcc)); return float(mcc[k]),float(ths[k])
def sv(df,col,w):
    if w<=0: return df[col].values
    d=pd.DataFrame({'p':df[col].values,'gp':df.gp.values,'p1':df.p1.values,'p2':df.p2.values,'s':df.step.values})
    d=d.sort_values(['gp','p1','p2','s']); g=d.groupby(['gp','p1','p2'],sort=False)['p']
    r=g.rolling(w,center=True,min_periods=1).mean(); r.index=r.index.droplevel([0,1,2])
    return r.reindex(d.index).sort_index().values
SRC='workspaces/canopus_pi/preds/oof_pp_gru_v4.parquet'
TSRC='workspaces/canopus_pi/preds/test_pp_gru_v4.parquet'
if not os.path.exists(SRC): print('missing',SRC); sys.exit(0)
o=pd.read_parquet('workspaces/capella_pi/oof_stack_mixv2.parquet')
d=pd.read_parquet(SRC)[['contact_id','pred']].rename(columns={'pred':'new'})
o=o.merge(d,on='contact_id',how='left')
sp=o.contact_id.str.split('_'); o['gp']=sp.str[0]+'_'+sp.str[1]; o['step']=sp.str[2].astype(int); o['p1']=sp.str[3]; o['p2']=sp.str[4]
pp=o.is_ground.values==0
best=(0.0,-1,0.5,0)
for a in [0.0,0.05,0.1,0.15,0.2,0.3]:
    o['p']=o.stack_oof.values+np.where(pp,a*logit(o['new'].fillna(1e-5).values),0.0)
    b=(-1,0.5,0)
    for w in [0,1,3,5]:
        p=sv(o[pp],'p',w); mcc,th=best_mcc(o.contact.values[pp],p)
        if mcc>b[0]: b=(mcc,th,w)
    print('alpha',a,'pp mcc',round(b[0],6),'w',b[2])
    if b[0]>best[1]: best=(a,b[0],b[1],b[2])
a=best[0]; print('best alpha',best)
o['stack_oof']=o.stack_oof.values+np.where(pp,a*logit(o['new'].fillna(1e-5).values),0.0)
o[['contact_id','fold','contact','is_ground','stack_oof']].to_parquet('workspaces/capella_pi/oof_stack_mixv3.parquet',index=False)
t=pd.read_parquet('workspaces/capella_pi/test_stack_mixv2.parquet')
dt=pd.read_parquet(TSRC)[['contact_id','pred']].rename(columns={'pred':'new'})
t=t.merge(dt,on='contact_id',how='left')
igt=t.contact_id.str.endswith('_G').values
t['stack_pred']=t.stack_pred.values+np.where(~igt,a*logit(t['new'].fillna(1e-5).values),0.0)
t[['contact_id','stack_pred']].to_parquet('workspaces/capella_pi/test_stack_mixv3.parquet',index=False)
json.dump({'alpha_pp_gru_v4':a},open('workspaces/capella_pi/mixv3_weights.json','w'))
print('saved mixv3')
