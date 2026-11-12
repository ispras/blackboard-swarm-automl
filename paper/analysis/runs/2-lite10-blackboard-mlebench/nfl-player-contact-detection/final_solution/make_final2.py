import numpy as np, pandas as pd, json, os, sys
sys.path.insert(0, '.')
from postprocess_search import smooth
from final_blend import joint_mcc

R='.'; A='../achernar_sigma'; C='../castor_alpha'
pw=json.load(open('pair_weights_search.json'))
names=pw['names']; w=np.array(pw['weights'])
print('pair weights',dict(zip(names,np.round(w,4))))

def mcc_curve(y,p):
    idx=np.argsort(-p); ys=y[idx].astype(np.float64)
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); pos=ys.sum(); neg=len(ys)-pos
    fn=pos-tp; tn=neg-fp; den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    with np.errstate(divide='ignore',invalid='ignore'):
        m=np.where(den>0,(tp*tn-fp*fn)/den,-1)
    k=int(np.argmax(m)); return float(m[k]), float(np.sort(p)[::-1][k])

pref=pd.read_parquet('oof_pair.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step','contact','fold']]
yp=pref.contact.values.astype(np.int8)
def load_oof(path,col):
    return pref[['contact_id']].merge(pd.read_parquet(path)[['contact_id',col]],on='contact_id',how='left')[col].values.astype(np.float64)
seq=np.mean([load_oof(f'oof_pair_seq_s{i}.parquet','prob') for i in range(3)],axis=0)
P=np.stack([load_oof('oof_pair.parquet','prob'),load_oof('oof_pair_h.parquet','prob'),
            load_oof(f'{A}/oof_pair_lgbv1.parquet','pred'),load_oof(f'{A}/oof_pair_lgbv2.parquet','pred'),
            load_oof(f'{A}/oof_pair_gstack.parquet','pred'),seq],axis=1)
p_pair=P@w
pp=pref.copy(); pp['_p']=p_pair
dsp,sp=smooth(pp,p_pair,3); sp=sp.astype(np.float64); yps=dsp.contact.values.astype(np.int8)
print('PAIR MCC',mcc_curve(yps,sp),'rate',float((sp>=mcc_curve(yps,sp)[1]).mean()))

# ground
gcfg=json.load(open(f'{C}/final_eff_blend.json'))
gmap={'track':('oof_ground.parquet','prob'),'ach2':(f'{A}/oof_ground_lgbv2.parquet','pred'),
      'gfus':(f'{A}/oof_ground_gfus.parquet','pred'),'ctab':(f'{C}/ground_oof_tab.parquet','oof'),
      'cvid':(f'{C}/ground_oof_video.parquet','pred'),'ceff':(f'{C}/ground_oof_video_eff.parquet','pred'),
      'stackr':('ground_oof_stack_r.parquet','prob'),'gstack':(f'{A}/oof_ground_gstack.parquet','pred')}
gref=pd.read_parquet('oof_ground.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step','contact']]
yg=gref.contact.values.astype(np.int8)
G={}
for n,(p,c) in gmap.items():
    G[n]=gref[['contact_id']].merge(pd.read_parquet(p)[['contact_id',c]],on='contact_id',how='left')[c].values.astype(np.float64)
p_ground=sum(wi*G[n] for n,wi in zip(gcfg['ground_models'],gcfg['ground_weights']))
gg=gref.copy(); gg['_p']=p_ground
dsg,sg=smooth(gg,p_ground,15); sg=sg.astype(np.float64); ygs=dsg.contact.values.astype(np.int8)
mg= mcc_curve(ygs,sg); print('GROUND MCC',mg)
r=joint_mcc(sp,yps,sg,ygs,nq=240); print('JOINT',r)

# test
te_pair=pd.read_parquet('testpred_pair.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']]
test_specs=[('track','testpred_pair.parquet','prob'),('pairh','testpred_pair_h.parquet','prob'),
            ('ach1',f'{A}/testpred_pair_lgbv1.parquet','pred'),('ach2',f'{A}/testpred_pair_lgbv2.parquet','pred'),
            ('gstack',f'{A}/testpred_pair_gstack.parquet','pred')]
for n,p,c in test_specs:
    te_pair=te_pair.merge(pd.read_parquet(p)[['contact_id',c]].rename(columns={c:n}),on='contact_id',how='left')
# seq test avg
st=None
for i in range(3):
    t=pd.read_parquet(f'testpred_pair_seq_s{i}.parquet')[['contact_id','prob']].rename(columns={'prob':f's{i}'})
    st=t if st is None else st.merge(t,on='contact_id')
st['seq']=st[['s0','s1','s2']].mean(axis=1)
te_pair=te_pair.merge(st[['contact_id','seq']],on='contact_id',how='left')
p_pair_test=sum(wi*te_pair[n].values for n,wi in zip(names,w))
te_pair['_p']=p_pair_test
dsp_t,spt=smooth(te_pair,p_pair_test,3)

te_g=pd.read_parquet('testpred_ground.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']]
gtest_map={'track':('testpred_ground.parquet','prob'),'ach2':(f'{A}/testpred_ground_lgbv2.parquet','pred'),
      'gfus':(f'{A}/testpred_ground_gfus.parquet','pred'),'ctab':(f'{C}/ground_test_tab.parquet','pred'),
      'cvid':(f'{C}/ground_test_video.parquet','pred'),'ceff':(f'{C}/ground_test_video_eff.parquet','pred'),
      'stackr':('ground_test_stack_r.parquet','prob'),'gstack':(f'{A}/testpred_ground_gstack.parquet','pred')}
for n,(p,c) in gtest_map.items():
    te_g=te_g.merge(pd.read_parquet(p)[['contact_id',c]].rename(columns={c:n}),on='contact_id',how='left')
p_g_test=sum(wi*te_g[n].values for n,wi in zip(gcfg['ground_models'],gcfg['ground_weights']))
te_g['_p']=p_g_test
dsg_t,sgt=smooth(te_g,p_g_test,15)

pair_out=pd.DataFrame({'contact_id':dsp_t.contact_id.values,'prob':spt})
ground_out=pd.DataFrame({'contact_id':dsg_t.contact_id.values,'prob':sgt})
sub=pd.read_csv('../../data/sample_submission.csv')
sub=sub.merge(pd.concat([pair_out,ground_out]),on='contact_id',how='left')
tgt_p=0.0120; tgt_g=0.0450
thr_p=float(np.quantile(pair_out.prob.values,1-tgt_p)); thr_g=float(np.quantile(ground_out.prob.values,1-tgt_g))
sub['contact']=(sub.prob.fillna(0)>=np.where(sub.contact_id.str.endswith('_G'),thr_g,thr_p)).astype(int)
sub[['contact_id','contact']].to_csv('submission_final2.csv',index=False)
print('test pos',sub.contact.mean(),'pair',sub[~sub.contact_id.str.endswith('_G')].contact.mean(),'ground',sub[sub.contact_id.str.endswith('_G')].contact.mean())
json.dump({'pair_names':names,'pair_weights':[float(x) for x in w],'wp':3,'thr_pair':mcc_curve(yps,sp)[1],
           'ground_models':gcfg['ground_models'],'ground_weights':gcfg['ground_weights'],'wg':15,'thr_ground':mg[1],
           'pair_mcc':mcc_curve(yps,sp)[0],'ground_mcc':mg[0],'joint_mcc':r[0],
           'rate_targets':{'pair':tgt_p,'ground':tgt_g},'test_pos_rate':float(sub.contact.mean())},
          open('blend_final2.json','w'),indent=2)
