import os, gc, time, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, roc_auc_score

W='workspaces/capella_pi'; P='workspaces/canopus_pi/preds'; A='workspaces/alkaid_psi'; ROOTP='.'
CACHE=f'{W}/cache_v3'; os.makedirs(CACHE,exist_ok=True)
t0=time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]',*a,flush=True)
def logit(p,eps=1e-6):
    p=np.clip(np.asarray(p,dtype=np.float64),eps,1-eps); return np.log(p/(1-p)).astype(np.float32)

RAWCOLS=['pair_dist','pair_dist_diff','h_min_cdist_norm','s_iou','e_iou','s_both','e_both',
         's_cdist_norm','e_cdist_norm','closing_speed','dir_diff','rel_vel','same_team',
         'p1_speed','p2_speed','p1_acceleration','p2_acceleration','p1_sa','p2_sa',
         'p1_nn_dist','p2_nn_dist','p1_n_within_2','p2_n_within_2',
         'p1_s_top','p1_e_top','p1_s_cy','p1_e_cy','p1_s_height','p1_e_height',
         's_p1_low','e_p1_low','s_p1_only','e_p1_only','p1_s_area','p1_e_area']

def srcs(split):
    """return list of (name, path, col) for probability sources"""
    if split=='train':
        L=[('p_pp',f'{P}/ens_oof_probs_v4.parquet','p_pp'),('p_gr',f'{P}/ens_oof_probs_v4.parquet','p_gr'),
           ('mine',f'{W}/oof_lgbm.parquet','pred'),('cnn',f'{A}/oof_cnn_cand_v2m.parquet','pred'),
           ('gcnn',f'{A}/oof_ground_cnn_v1m.parquet','pred'),
           ('ggru',f'{P}/oof_ground_gru_v1.parquet','pred')]
        for v in [1,2,3,4]: L.append((f'cpp{v}',f'{P}/oof_pp_v{v}.parquet','pred'))
        for v in [1,2,3]: L.append((f'cgr{v}',f'{P}/oof_ground_v{v}.parquet','pred'))
    else:
        L=[('p_pp',f'{P}/ens_test_probs_v4.parquet','p_pp'),('p_gr',f'{P}/ens_test_probs_v4.parquet','p_gr'),
           ('mine',f'{W}/test_lgbm.parquet','pred'),('cnn',f'{A}/test_cnn_cand_v2m.parquet','pred'),
           ('gcnn',f'{A}/test_ground_cnn_v1m.parquet','pred'),
           ('ggru',f'{P}/test_ground_gru_v1.parquet','pred')]
        for v in [1,2,3,4]: L.append((f'cpp{v}',f'{P}/test_pp_v{v}.parquet','pred'))
        for v in [1,2,3]: L.append((f'cgr{v}',f'{P}/test_ground_v{v}.parquet','pred'))
    # optional newer artifacts
    if split=='train':
        L+=[('gcnnB',f'{A}/oof_ground_cnn_vB1m.parquet','pred'),
            ('gcnnC',f'{A}/oof_ground_cnn_vC1m.parquet','pred'),
            ('ppgru',f'{P}/oof_pp_gru_v1.parquet','pred'),
            ('ppgru2',f'{P}/oof_pp_gru_v2.parquet','pred'),
            ('ppgru3',f'{P}/oof_pp_gru_v3.parquet','pred'),
            ('ppxgb',f'{P}/oof_pp_xgb1.parquet','pred'),
            ('grxgb',f'{P}/oof_ground_xgb1.parquet','pred'),
            ('cgr5',f'{P}/oof_ground_v5.parquet','pred'),
            ('gextra',f'{P}/oof_ground_extra_v1.parquet','pred'),
            ('cpp5',f'{P}/oof_pp_v5.parquet','pred'),
            ('ggru2',f'{P}/oof_ground_gru_v2.parquet','pred'),
            ('ggru3',f'{P}/oof_ground_gru_v3.parquet','pred'),
            ('ggru4',f'{P}/oof_ground_gru_v4.parquet','pred'),
            ('gcnnD',f'{A}/oof_ground_cnn_vD1m.parquet','pred'),
            ('cgr4',f'{P}/oof_ground_v4.parquet','pred')]
    else:
        L+=[('gcnnB',f'{A}/test_ground_cnn_vB1m.parquet','pred'),
            ('gcnnC',f'{A}/test_ground_cnn_vC1m.parquet','pred'),
            ('ppgru',f'{P}/test_pp_gru_v1.parquet','pred'),
            ('ppgru2',f'{P}/test_pp_gru_v2.parquet','pred'),
            ('ppgru3',f'{P}/test_pp_gru_v3.parquet','pred'),
            ('ppxgb',f'{P}/test_pp_xgb1.parquet','pred'),
            ('grxgb',f'{P}/test_ground_xgb1.parquet','pred'),
            ('cgr5',f'{P}/test_ground_v5.parquet','pred'),
            ('gextra',f'{P}/test_ground_extra_v1.parquet','pred'),
            ('cpp5',f'{P}/test_pp_v5.parquet','pred'),
            ('ggru2',f'{P}/test_ground_gru_v2.parquet','pred'),
            ('ggru3',f'{P}/test_ground_gru_v3.parquet','pred'),
            ('ggru4',f'{P}/test_ground_gru_v4.parquet','pred'),
            ('gcnnD',f'{A}/test_ground_cnn_vD1m.parquet','pred'),
            ('cgr4',f'{P}/test_ground_v4.parquet','pred')]
    return [(n,p,c) for (n,p,c) in L if os.path.exists(p)]

def load_base_table(split):
    if split=='train':
        lab=pd.read_csv('data/train_labels.csv',usecols=['game_play','step','nfl_player_id_1','nfl_player_id_2','contact'])
        base=pd.DataFrame({'gp':lab.game_play.astype(str),'step':lab.step.astype(np.int16),
                           'p1':lab.nfl_player_id_1.astype(np.int64),'p2':lab.nfl_player_id_2.astype(str),
                           'contact':lab.contact.astype(np.int8)})
    else:
        ss=pd.read_csv('data/sample_submission.csv'); sp=ss.contact_id.str.split('_')
        base=pd.DataFrame({'gp':sp.str[0]+'_'+sp.str[1],'step':sp.str[2].astype(np.int16),
                           'p1':sp.str[3].astype(np.int64),'p2':sp.str[4],'contact':np.nan})
    base['is_ground']=(base.p2=='G').astype(np.int8)
    base['contact_id']=(base.gp+'_'+base.step.astype(str)+'_'+base.p1.astype(str)+'_'+base.p2)
    if split=='train':
        folds=pd.read_csv(f'{A}/cv_folds.csv').rename(columns={'game_play':'gp'})
        base=base.merge(folds,on='gp',how='left'); base['fold']=base['fold'].astype(np.int8)
    else: base['fold']=np.int8(-1)
    return base

def build(split):
    log(split,'base'); df=load_base_table(split)
    raw=pd.read_parquet(f'{W}/{split}_feat.parquet',columns=RAWCOLS+['game_play','step','nfl_player_id_1','nfl_player_id_2'])
    raw['contact_id']=(raw.game_play.astype(str)+'_'+raw.step.astype(str)+'_'+raw.nfl_player_id_1.astype(str)+'_'+raw.nfl_player_id_2.astype(str))
    raw=raw.drop(columns=['game_play','step','nfl_player_id_1','nfl_player_id_2'])
    df=df.merge(raw,on='contact_id',how='left'); del raw; gc.collect()
    ge_path=f'{ROOTP}/workspaces/canopus_pi/features/ground_extra.parquet'
    if os.path.exists(ge_path) and os.environ.get('NO_EXTRA')!='1':
        ge=pd.read_parquet(ge_path)
        df=df.merge(ge,on='contact_id',how='left'); del ge; gc.collect(); log(split,'ground_extra merged')
    used=[]
    for name,path,col in srcs(split):
        d=pd.read_parquet(path)
        if name in d.columns:
            d=d[['contact_id',name]].copy()
        else:
            d=d[['contact_id',col]].rename(columns={col:name})
        df=df.merge(d,on='contact_id',how='left'); used.append(name)
    log(split,'probs used',used)
    for c in used:
        df[c]=df[c].astype(np.float32); df[c+'_l']=logit(df[c].fillna(1e-5).values)
    # temporal
    keys=['gp','p1','p2']; df=df.sort_values(keys+['step']).reset_index(drop=True)
    gb=df.groupby(keys,sort=False,observed=True)
    for c in [u+'_l' for u in used]:
        g=gb[c]
        for k in [1,2,3,5]:
            df[f'{c}_lag{k}']=g.shift(k).astype(np.float32); df[f'{c}_lead{k}']=g.shift(-k).astype(np.float32)
        for w in [5,11,21]:
            r=gb[c].rolling(w,center=True,min_periods=1).mean(); r.index=r.index.droplevel([0,1,2])
            df[f'{c}_r{w}']=r.reindex(df.index).astype(np.float32)
        r=gb[c].rolling(11,center=True,min_periods=1).max(); r.index=r.index.droplevel([0,1,2])
        df[f'{c}_r11max']=r.reindex(df.index).astype(np.float32)
        r=gb[c].rolling(21,center=True,min_periods=1).std(); r.index=r.index.droplevel([0,1,2])
        df[f'{c}_r21std']=r.reindex(df.index).astype(np.float32)
    # disagreement features (task specific)
    pp_names=[u for u in used if u in ('p_pp','mine','cnn','cpp1','cpp2','cpp3','cpp4','cpp5','ppgru','ppgru2','ppgru3','ppxgb')]
    g_names=[u for u in used if u in ('p_gr','mine','gcnn','gcnnB','gcnnC','gcnnD','ggru','ggru2','ggru3','ggru4','cgr1','cgr2','cgr3','cgr4','cgr5','gextra','grxgb')]
    def agg(cols,prefix):
        if not cols: return
        M=np.column_stack([df[c+'_l'].values for c in cols])
        with np.errstate(all='ignore'):
            df[prefix+'_lstd']=np.nanstd(M,axis=1).astype(np.float32)
            df[prefix+'_lrange']=(np.nanmax(M,axis=1)-np.nanmin(M,axis=1)).astype(np.float32)
            df[prefix+'_lmean']=np.nanmean(M,axis=1).astype(np.float32)
        if 'cnn_l' in dir() or True:
            pass
    agg(pp_names,'dis_pp'); agg(g_names,'dis_g')
    return df,used

def smooth(pred,gp,p1,p2,step,w):
    if w<=0: return np.asarray(pred)
    d=pd.DataFrame({'p':np.asarray(pred),'gp':np.asarray(gp),'p1':np.asarray(p1),'p2':np.asarray(p2),'s':np.asarray(step)})
    d=d.sort_values(['gp','p1','p2','s']); g=d.groupby(['gp','p1','p2'],sort=False,observed=True)['p']
    r=g.rolling(w,center=True,min_periods=1).mean(); r.index=r.index.droplevel([0,1,2])
    return r.reindex(d.index).sort_index().values

def best_mcc(y,p):
    ths=np.quantile(p,np.linspace(0.5,0.9995,300)); order=np.argsort(-p); ps=p[order]; ys=np.asarray(y)[order]
    cp=np.cumsum(ys,dtype=np.float64); cn=np.cumsum(1-ys,dtype=np.float64)
    Pos=int(ys.sum()); Neg=len(ys)-Pos
    idx=np.clip(np.searchsorted(-ps,-ths,side='left'),0,len(ys))
    tp=np.where(idx>0,cp[np.maximum(idx-1,0)],0.0); fp=np.where(idx>0,cn[np.maximum(idx-1,0)],0.0)
    fn=Pos-tp; tn=Neg-fp; den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+1e-12
    mcc=(tp*tn-fp*fn)/den; k=int(np.argmax(mcc)); return float(mcc[k]),float(ths[k])

def evaluate(df,prob,label=''):
    ig=df.is_ground.values==1; res={}
    for task,mask in [('pp',~ig),('ground',ig)]:
        best=(-1,0.5,0)
        for w in [0,1,3,5,7,9]:
            p=smooth(prob[mask],df.gp.values[mask],df.p1.values[mask],df.p2.values[mask],df.step.values[mask],w)
            mcc,th=best_mcc(df.contact.values[mask],p)
            if mcc>best[0]: best=(mcc,th,w)
        res[task]=best; log(label,task,'mcc',round(best[0],5),'th',round(best[1],5),'w',best[2])
    mpp,thpp,wpp=res['pp']; mg,thg,wg=res['ground']
    ppp=smooth(prob[~ig],df.gp.values[~ig],df.p1.values[~ig],df.p2.values[~ig],df.step.values[~ig],wpp)
    pg=smooth(prob[ig],df.gp.values[ig],df.p1.values[ig],df.p2.values[ig],df.step.values[ig],wg)
    pred=np.concatenate([ppp>thpp,pg>thg]); y=np.concatenate([df.contact.values[~ig],df.contact.values[ig]])
    g=matthews_corrcoef(y,pred); log(label,'GLOBAL',round(g,5)); res['global']=(g,thpp,thg,wpp,wg)
    return res

def train_task(X,y,fv,groups,Xt,params,neg_frac,seeds=(7,)):
    oof=np.zeros(len(X),np.float32); tp=np.zeros(len(Xt),np.float32); iters=[]
    for seed in seeds:
      p=dict(params); p['seed']=seed
      p['bagging_seed']=seed+1; p['feature_fraction_seed']=seed+2
      oof_s=np.zeros(len(X),np.float32); tp_s=np.zeros(len(Xt),np.float32); it=[]
      for f in range(5):
        va=np.where(fv==f)[0]; tri_all=np.where(fv!=f)[0]
        rng=np.random.RandomState(500+f)
        neg=tri_all[y[tri_all]==0]; pos=tri_all[y[tri_all]==1]
        if neg_frac<1.0:
            tri=np.sort(np.concatenate([pos,rng.choice(neg,int(len(neg)*neg_frac),replace=False)]))
        else: tri=tri_all
        dtr=lgb.Dataset(X.iloc[tri],label=y[tri],free_raw_data=True)
        dva=lgb.Dataset(X.iloc[va],label=y[va],reference=dtr,free_raw_data=True)
        m=lgb.train(p,dtr,num_boost_round=2500,valid_sets=[dva],callbacks=[lgb.early_stopping(100,verbose=False)])
        it.append(m.best_iteration); oof_s[va]=m.predict(X.iloc[va],num_iteration=m.best_iteration).astype(np.float32)
        tp_s+=m.predict(Xt,num_iteration=m.best_iteration).astype(np.float32)/5.0
        log('  seed',seed,'fold',f,'it',m.best_iteration,'auc',round(float(roc_auc_score(y[va],oof_s[va])),5))
        del dtr,dva,m; gc.collect()
      oof+=oof_s/len(seeds); tp+=tp_s/len(seeds); iters.append(it)
    return oof,tp,iters

EXTRA_FEATS=['run_low0.5','fut_low0.5','run_low1.5','fut_low1.5','run_low2.5','fut_low2.5',
 'speed_c5_min','speed_c5_max','speed_c5_mean','sa_c5_min','sa_c5_max','sa_c5_mean',
 'acceleration_c5_min','acceleration_c5_max','acceleration_c5_mean','distance_c5_min',
 'distance_c5_max','distance_c5_mean','dist_fut5_sum','dist_past5_sum','rank_speed','rank_sa',
 'rank_acceleration','rank_distance','sid_top_rank2','end_top_rank2','sid_bottom_rank2',
 'end_bottom_rank2','sid_h_rank2','end_h_rank2']
COMMONBASE=['is_ground','step','pair_dist','pair_dist_diff','h_min_cdist_norm','s_cdist_norm','e_cdist_norm',
        's_iou','e_iou','s_both','e_both','closing_speed','dir_diff','rel_vel','same_team',
        'p1_speed','p2_speed','p1_acceleration','p2_acceleration','p1_sa','p2_sa',
        'p1_nn_dist','p2_nn_dist','p1_n_within_2','p2_n_within_2']
COMMON=COMMONBASE+([] if os.environ.get('NO_EXTRA')=='1' else EXTRA_FEATS)
GHELM=['p1_s_top','p1_e_top','p1_s_cy','p1_e_cy','p1_s_height','p1_e_height',
       's_p1_low','e_p1_low','s_p1_only','e_p1_only','p1_s_area','p1_e_area']

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--rebuild',action='store_true'); ap.add_argument('--seeds',type=int,default=1); ap.add_argument('--tag',default='v3')
    args=ap.parse_args()
    mt=os.path.join(CACHE,'meta.json')
    if (not args.rebuild) and os.path.exists(mt) and os.path.exists(f'{CACHE}/train.parquet'):
        tr=pd.read_parquet(f'{CACHE}/train.parquet'); te=pd.read_parquet(f'{CACHE}/test.parquet')
        md=json.load(open(mt)); used=md['used']; usedt=md['usedt']; log('loaded cache')
    else:
        tr,used=build('train'); te,usedt=build('test')
        tr.to_parquet(f'{CACHE}/train.parquet'); te.to_parquet(f'{CACHE}/test.parquet')
        json.dump({'used':used,'usedt':usedt},open(mt,'w'))
    pp_names=[u for u in used if u in ('p_pp','mine','cnn','cpp1','cpp2','cpp3','cpp4','cpp5','ppgru','ppgru2','ppgru3','ppxgb')]
    g_names=[u for u in used if u in ('p_gr','mine','gcnn','gcnnB','gcnnC','gcnnD','ggru','ggru2','ggru3','ggru4','cgr1','cgr2','cgr3','cgr4','cgr5','gextra','grxgb')]
    def feats_for(names,task):
        fs=[n+'_l' for n in names]
        fs+= [c for c in tr.columns if any(c.startswith(n+'_l_') for n in names)]
        fs+= COMMON + (GHELM if task=='g' else [])
        fs+= [f'dis_{task}_lstd',f'dis_{task}_lrange',f'dis_{task}_lmean']
        return list(dict.fromkeys([c for c in fs if c in te.columns and c in tr.columns]))
    pp_feats=feats_for(pp_names,'pp'); g_feats=feats_for(g_names,'g')
    ig=tr.is_ground.values==1
    params_pp=dict(objective='binary',metric='auc',learning_rate=0.04,num_leaves=64,min_data_in_leaf=200,
                   feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=10.0,num_threads=12,
                   verbosity=-1,max_bin=127,seed=7)
    params_g=dict(objective='binary',metric='auc',learning_rate=0.035,num_leaves=48,min_data_in_leaf=150,
                  feature_fraction=0.65,bagging_fraction=0.8,bagging_freq=1,lambda_l2=20.0,num_threads=12,
                  verbosity=-1,max_bin=127,seed=11)
    log('pp feats',len(pp_feats),'g feats',len(g_feats))
    trp=tr[~ig].reset_index(drop=True); tep=te[te.is_ground.values==0].reset_index(drop=True)
    Xp=trp[pp_feats].astype(np.float32).replace([np.inf,-np.inf],np.nan)
    Xtp=tep[pp_feats].astype(np.float32).replace([np.inf,-np.inf],np.nan)
    SEEDS=tuple(7+13*i for i in range(args.seeds))
    oof_p,tp_p,itp=train_task(Xp,trp.contact.values.astype(np.int8),trp.fold.values,trp.gp.values,Xtp,params_pp,0.30,SEEDS)
    trg=tr[ig].reset_index(drop=True); teg=te[te.is_ground.values==1].reset_index(drop=True)
    Xg=trg[g_feats].astype(np.float32).replace([np.inf,-np.inf],np.nan)
    Xtg=teg[g_feats].astype(np.float32).replace([np.inf,-np.inf],np.nan)
    oof_g,tp_g,itg=train_task(Xg,trg.contact.values.astype(np.int8),trg.fold.values,trg.gp.values,Xtg,params_g,1.0,SEEDS)
    oof=np.empty(len(tr),np.float32); oof[~ig]=oof_p; oof[ig]=oof_g
    tp=np.empty(len(te),np.float32); mte=te.is_ground.values==0; tp[mte]=tp_p; tp[~mte]=tp_g
    base=np.where(ig,tr.p_gr.values,tr.p_pp.values); base=np.nan_to_num(base,nan=0.0)
    rb=evaluate(tr,base,'base'); rs=evaluate(tr,oof,'stackv3')
    tr[['contact_id','fold','contact','is_ground']].assign(stack_oof=oof).to_parquet(f'{W}/oof_stack_{args.tag}.parquet',index=False)
    te[['contact_id']].assign(stack_pred=tp).to_parquet(f'{W}/test_stack_{args.tag}.parquet',index=False)
    json.dump({'base':rb,'stackv3':rs,'iters_pp':itp,'iters_g':itg,'used':used,'usedt':usedt,'npp':len(pp_feats),'ng':len(g_feats)},
              open(f'{W}/stack_{args.tag}_metrics.json','w'),indent=1,default=float)
    log('saved v3')

if __name__=='__main__': main()
