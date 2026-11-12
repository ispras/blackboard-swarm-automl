import numpy as np, pandas as pd, json, os, sys
sys.path.insert(0, '.')
from postprocess_search import smooth, KEYS
from final_blend import joint_mcc

R='.'; A='../achernar_sigma'; C='../castor_alpha'

def mcc_curve(y,p):
    idx=np.argsort(-p); ys=y[idx].astype(np.float64)
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); pos=ys.sum(); neg=len(ys)-pos
    fn=pos-tp; tn=neg-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    with np.errstate(divide='ignore',invalid='ignore'):
        m=np.where(den>0,(tp*tn-fp*fn)/den,-1)
    k=int(np.argmax(m)); return float(m[k]), float(np.sort(p)[::-1][k])

# ---------- PAIR OOF ----------
pref=pd.read_parquet('oof_pair.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step','contact','fold']]
yp=pref.contact.values.astype(np.int8)
def p_oof(path,col):
    return pref[['contact_id']].merge(pd.read_parquet(path)[['contact_id',col]],on='contact_id',how='left')[col].values.astype(np.float64)
def p_test(path,col):
    return pd.read_parquet(path)[['contact_id',col]].rename(columns={col:path.split('/')[-1]+'_'+col})
pair_specs=[('track','oof_pair.parquet','prob','testpred_pair.parquet','prob',0.037),
            ('pairh','oof_pair_h.parquet','prob','testpred_pair_h.parquet','prob',0.130),
            ('ach1',f'{A}/oof_pair_lgbv1.parquet','pred',f'{A}/testpred_pair_lgbv1.parquet','pred',0.004),
            ('ach2',f'{A}/oof_pair_lgbv2.parquet','pred',f'{A}/testpred_pair_lgbv2.parquet','pred',0.023),
            ('gstack',f'{A}/oof_pair_gstack.parquet','pred',f'{A}/testpred_pair_gstack.parquet','pred',0.806)]
pair_oof={}; pair_test={}
for n,op,oc,tp,tc,w in pair_specs:
    pair_oof[n]=p_oof(op,oc); pair_test[n]=pd.read_parquet(tp)[['contact_id',tc]].rename(columns={tc:n})
p_base=sum(w*pair_oof[n] for n,_,_,_,_,w in pair_specs)
ens=pd.read_parquet(f'{C}/pair_oof_ens_eff.parquet')
ref=pref[['contact_id']].merge(ens[['contact_id','pred']],on='contact_id',how='left')['pred'].values
print('pair base reprod corr',np.corrcoef(p_base,ref)[0,1],'maxdiff',np.nanmax(np.abs(p_base-ref)))

seqs=[f'oof_pair_seq_s{i}.parquet' for i in range(3) if os.path.exists(f'oof_pair_seq_s{i}.parquet')]
seq_oof=np.mean([p_oof(s,'prob') for s in seqs],axis=0)
seq_test=None
for s in seqs:
    t=pd.read_parquet(s.replace('oof_pair_seq','testpred_pair_seq'))[['contact_id','prob']].rename(columns={'prob':'s'+s})
    seq_test=t if seq_test is None else seq_test.merge(t,on='contact_id')
seq_test['seq']=seq_test[[c for c in seq_test.columns if c!='contact_id']].mean(axis=1)
print('seq seeds',seqs,'seq oof mcc',mcc_curve(yp,seq_oof)[0])

best=(-1,None)
for w in np.arange(0,0.401,0.02):
    p=(1-w)*p_base+w*seq_oof
    for wp in [3,5]:
        if wp==1: ps=p
        else:
            ds,ps=smooth(pref.assign(_p=p),p,wp); ps=ps.astype(np.float64)
        m,t=mcc_curve(yp,ps)
        if m>best[0]: best=(m,(round(w,3),wp,t))
    print('w',round(w,3),'best',best,flush=True)
print('PAIR BEST',best)
w_seq,wp,tp=best[1]
p_pair=(1-w_seq)*p_base+w_seq*seq_oof
ds,sp=smooth(pref.assign(_p=p_pair),p_pair,wp); sp=sp.astype(np.float64); yps=ds.contact.values.astype(np.int8)
print('pair rate at thr',float((sp>=tp).mean()))

# ---------- GROUND OOF ----------
gref=pd.read_parquet('oof_ground.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step','contact','fold']]
yg=gref.contact.values.astype(np.int8)
gcfg=json.load(open(f'{C}/final_eff_blend.json'))
gmap={'track':('oof_ground.parquet','prob',f'{R}/testpred_ground.parquet','prob'),
      'ach2':(f'{A}/oof_ground_lgbv2.parquet','pred',f'{A}/testpred_ground_lgbv2.parquet','pred'),
      'gfus':(f'{A}/oof_ground_gfus.parquet','pred',f'{A}/testpred_ground_gfus.parquet','pred'),
      'ctab':(f'{C}/ground_oof_tab.parquet','oof',f'{C}/ground_test_tab.parquet','pred'),
      'cvid':(f'{C}/ground_oof_video.parquet','pred',f'{C}/ground_test_video.parquet','pred'),
      'ceff':(f'{C}/ground_oof_video_eff.parquet','pred',f'{C}/ground_test_video_eff.parquet','pred'),
      'stackr':('ground_oof_stack_r.parquet','prob','ground_test_stack_r.parquet','prob'),
      'gstack':(f'{A}/oof_ground_gstack.parquet','pred',f'{A}/testpred_ground_gstack.parquet','pred')}
g_oof={}; g_test={}
for n,(op,oc,tpth,tc) in gmap.items():
    g_oof[n]=gref[['contact_id']].merge(pd.read_parquet(op)[['contact_id',oc]],on='contact_id',how='left')[oc].values.astype(np.float64)
    g_test[n]=pd.read_parquet(tpth)[['contact_id',tc]].rename(columns={tc:n})
p_ground=sum(w*g_oof[n] for n,w in zip(gcfg['ground_models'],gcfg['ground_weights']))
ensg=gref[['contact_id']].merge(pd.read_parquet(f'{C}/ground_oof_ens_eff.parquet')[['contact_id','pred']],on='contact_id',how='left')['pred'].values
print('ground reprod corr',np.corrcoef(p_ground,ensg)[0,1],'maxdiff',np.nanmax(np.abs(p_ground-ensg)))
gg=gref.copy(); gg['_p']=p_ground
dsg,sg=smooth(gg,p_ground,15); sg=sg.astype(np.float64); ygs=dsg.contact.values.astype(np.int8)
mg,tg=mcc_curve(ygs,sg); print('GROUND MCC',mg,'thr',tg,'rate',float((sg>=tg).mean()))

# joint thresholds
r=joint_mcc(sp,yps,sg,ygs,nq=220)
print('JOINT',r)

# ---------- TEST ----------
base=None
for n,op,oc,tpt,tc,w in pair_specs:
    base=pair_test[n] if base is None else base.merge(pair_test[n],on='contact_id')
p_base_test=sum(w*base[n].values for n,_,_,_,_,w in pair_specs)
tp_df=pd.read_parquet('testpred_pair.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']]
tp_df=tp_df.merge(base[['contact_id']],on='contact_id')
tp_df['_p']=(1-w_seq)*p_base_test+w_seq*seq_test.set_index('contact_id').loc[tp_df.contact_id,'seq'].values
dsp,spt=smooth(tp_df,tp_df['_p'].values,wp)

gbase=None
for n in gcfg['ground_models']:
    gbase=g_test[n] if gbase is None else gbase.merge(g_test[n],on='contact_id')
g_base_test=sum(w*gbase[n].values for n,w in zip(gcfg['ground_models'],gcfg['ground_weights']))
tg_df=pd.read_parquet('testpred_ground.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']]
tg_df=tg_df.merge(gbase[['contact_id']],on='contact_id')
tg_df['_p']=g_base_test
dsg_t,sgt=smooth(tg_df,tg_df['_p'].values,15)

pair_out=pd.DataFrame({'contact_id':dsp.contact_id.values,'prob':spt})
ground_out=pd.DataFrame({'contact_id':dsg_t.contact_id.values,'prob':sgt})
sub=pd.read_csv('../../data/sample_submission.csv')
sub=sub.merge(pd.concat([pair_out,ground_out]),on='contact_id',how='left')
# rate match targets
tgt_p=0.0120; tgt_g=0.0450
thr_p=float(np.quantile(pair_out.prob.values,1-tgt_p)); thr_g=float(np.quantile(ground_out.prob.values,1-tgt_g))
sub['contact']=(sub.prob.fillna(0)>=np.where(sub.contact_id.str.endswith('_G'),thr_g,thr_p)).astype(int)
sub[['contact_id','contact']].to_csv('submission_eff.csv',index=False)
print('saved submission_eff.csv pos rate',sub.contact.mean(),'pair',sub[~sub.contact_id.str.endswith('_G')].contact.mean(),'ground',sub[sub.contact_id.str.endswith('_G')].contact.mean())
json.dump({'pair_weights':{n:w for n,_,_,_,_,w in pair_specs},'seq_weight':float(w_seq),'wp':wp,'thr_pair':tp,
           'ground_models':gcfg['ground_models'],'ground_weights':gcfg['ground_weights'],'wg':15,'thr_ground':tg,
           'joint_mcc':r[0],'joint_thr_pair':r[1],'joint_thr_ground':r[2],
           'rate_targets':{'pair':tgt_p,'ground':tgt_g},'test_pos_rate':float(sub.contact.mean())},
          open('blend_eff.json','w'),indent=2)
