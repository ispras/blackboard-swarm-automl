import pandas as pd, numpy as np, json, sys, time
from scipy.stats import rankdata
sys.path.insert(0,'workspaces/sirius_tau')
from lib_common import best_thr
OUT='workspaces/sirius_tau'; t0=time.time()
R=np.load(f'{OUT}/blend_R.npy'); RT=np.load(f'{OUT}/blend_RT.npy'); y=np.load(f'{OUT}/blend_y.npy')
names=json.load(open(f'{OUT}/blend_names.json'))
base=pd.read_parquet('workspaces/procyon_theta/oof_pred.parquet',columns=['contact_id'])
cid=base.contact_id.values
isg=np.array([s.endswith('_G') for s in cid])
sub=pd.read_csv('data/sample_submission.csv'); tcid=sub.contact_id.values
isgt=np.array([s.endswith('_G') for s in tcid])

def greedy(Rm,ym):
    bi=None; bm=-1
    for i in range(len(names)):
        m,_=best_thr(ym,Rm[i]); 
        if m>bm: bm=m; bi=i
    sel=[bi]; ws=Rm[bi].copy(); best=bm
    improved=True
    while improved:
        improved=False; cb=best; ci=-1
        for i in range(len(names)):
            m,_=best_thr(ym,ws+Rm[i])
            if m>cb+1e-5: cb=m; ci=i
        if ci>=0: sel.append(ci); ws=ws+Rm[ci]; best=cb; improved=True
    return sel,ws/len(sel),best

Rg=R[:,isg]; yg=y[isg]; Rp=R[:,~isg]; yp=y[~isg]
print('ground greedy',flush=True)
sg,bsg,mg=greedy(Rg,yg); print('ground sel',[names[i] for i in sg],'MCC',round(mg,4),flush=True)
print('pair greedy',flush=True)
sp,bsp,mp=greedy(Rp,yp); print('pair sel',[names[i] for i in sp],'MCC',round(mp,4),flush=True)
# combine: rank-normalize within type
og=np.zeros(len(y)); op=np.zeros(len(y))
og[isg]=rankdata(bsg)/isg.sum(); op[~isg]=rankdata(bsp)/(~isg).sum()
comb=np.where(isg,og,op)
def smooth_in(v,cids,w=5):
    parts=pd.Series(cids).str.rsplit('_',n=3,expand=True)
    df=pd.DataFrame({'gp':parts[0].values,'s':parts[1].values.astype(int),'p1':parts[2].values,'p2':parts[3].values,'v':v})
    df=df.sort_values(['gp','p1','p2','s'])
    sm=df.groupby(['gp','p1','p2'],sort=False)['v'].transform(lambda x:x.rolling(w,center=True,min_periods=1).mean()).values
    out=np.zeros(len(v)); out[df.index.values]=sm; return out
combs=smooth_in(comb,cid,5)
thr,mcc=best_thr(y,combs); q=(combs>=thr).mean()
print('COMBINED rank-norm RAW MCC',round(best_thr(y,comb)[1],5),flush=True)
print('COMBINED rank-norm SMOOTH MCC',round(mcc,5),'thr',round(thr,5),'q',round(q,4),flush=True)

# test
sgt=sum(RT[i] for i in sg)/len(sg); spt=sum(RT[i] for i in sp)/len(sp)
otg=np.zeros(len(tcid)); otp=np.zeros(len(tcid))
otg[isgt]=rankdata(sgt[isgt])/isgt.sum(); otp[~isgt]=rankdata(spt[~isgt])/(~isgt).sum()
combt=np.where(isgt,otg,otp)
def smooth(v,cids,w=5):
    parts=pd.Series(cids).str.rsplit('_',n=3,expand=True)
    df=pd.DataFrame({'gp':parts[0].values,'s':parts[1].values.astype(int),'p1':parts[2].values,'p2':parts[3].values,'v':v})
    df=df.sort_values(['gp','p1','p2','s'])
    sm=df.groupby(['gp','p1','p2'],sort=False)['v'].transform(lambda x:x.rolling(w,center=True,min_periods=1).mean()).values
    out=np.zeros(len(v)); out[df.index.values]=sm; return out
combt_s=smooth(combt,tcid,5)
thr_t=np.quantile(combt_s,1-q)
sub['contact']=(combt_s>=thr_t).astype(int)
sub[['contact_id','contact']].to_csv(f'{OUT}/sub_pertype.csv',index=False)
json.dump({'mcc':float(mcc),'q':float(q),'g':[names[i] for i in sg],'p':[names[i] for i in sp]},open(f'{OUT}/pertype_meta.json','w'))
print('saved sub_pertype posrate',sub.contact.mean(),'T',round(time.time()-t0),flush=True)
