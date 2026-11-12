import numpy as np, pandas as pd, json, sys
sys.path.insert(0,'.')
from postprocess_search import smooth
from final_blend import joint_mcc

R='.'; A='../achernar_sigma'; C='../castor_alpha'
pw=json.load(open('pair_weights_search.json')); names=pw['names']; w=np.array(pw['weights'])
pref=pd.read_parquet('oof_pair.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step','contact','fold']]
yp=pref.contact.values.astype(np.int8)
def lo(p,c): return pref[['contact_id']].merge(pd.read_parquet(p)[['contact_id',c]],on='contact_id',how='left')[c].values.astype(np.float64)
seq=np.mean([lo(f'oof_pair_seq_s{i}.parquet','prob') for i in range(5)],axis=0)
P=np.stack([lo('oof_pair.parquet','prob'),lo('oof_pair_h.parquet','prob'),lo(f'{A}/oof_pair_lgbv1.parquet','pred'),
            lo(f'{A}/oof_pair_lgbv2.parquet','pred'),lo(f'{A}/oof_pair_gstack.parquet','pred'),seq],axis=1)
p_pair=P@w
pp=pref.copy(); pp['_p']=p_pair
dsp,sp=smooth(pp,p_pair,3); sp=sp.astype(np.float64); yps=dsp.contact.values.astype(np.int8)

gb=pd.read_parquet('ground_oof_best.parquet')
yg=gb.contact.values.astype(np.int8); sg=gb.prob.values.astype(np.float64)
r=joint_mcc(sp,yps,sg,yg,nq=240); print('JOINT',r)

# test pair
te=pd.read_parquet('testpred_pair.parquet')[['contact_id','game_play','nfl_player_id_1','nfl_player_id_2','step']]
specs=[('track','testpred_pair.parquet','prob'),('pairh','testpred_pair_h.parquet','prob'),
       ('ach1',f'{A}/testpred_pair_lgbv1.parquet','pred'),('ach2',f'{A}/testpred_pair_lgbv2.parquet','pred'),
       ('gstack',f'{A}/testpred_pair_gstack.parquet','pred')]
for n,p,c in specs:
    te=te.merge(pd.read_parquet(p)[['contact_id',c]].rename(columns={c:n}),on='contact_id',how='left')
st=None
for i in range(5):
    t=pd.read_parquet(f'testpred_pair_seq_s{i}.parquet')[['contact_id','prob']].rename(columns={'prob':f's{i}'})
    st=t if st is None else st.merge(t,on='contact_id')
st['seq']=st[[f's{i}' for i in range(5)]].mean(axis=1)
te=te.merge(st[['contact_id','seq']],on='contact_id',how='left')
p_test=sum(wi*te[n].values for n,wi in zip(names,w)); te['_p']=p_test
dsp_t,spt=smooth(te,p_test,3)

gt=pd.read_parquet('ground_test_best.parquet')
pair_out=pd.DataFrame({'contact_id':dsp_t.contact_id.values,'prob':spt})
ground_out=pd.DataFrame({'contact_id':gt.contact_id.values,'prob':gt.prob.values})
sub=pd.read_csv('../../data/sample_submission.csv')
sub=sub.merge(pd.concat([pair_out,ground_out]),on='contact_id',how='left')
tgt_p=0.0120; tgt_g=0.0450
thr_p=float(np.quantile(pair_out.prob.values,1-tgt_p)); thr_g=float(np.quantile(ground_out.prob.values,1-tgt_g))
sub['contact']=(sub.prob.fillna(0)>=np.where(sub.contact_id.str.endswith('_G'),thr_g,thr_p)).astype(int)
sub[['contact_id','contact']].to_csv('submission_final3.csv',index=False)
print('pos',sub.contact.mean(),'pair',sub[~sub.contact_id.str.endswith('_G')].contact.mean(),'ground',sub[sub.contact_id.str.endswith('_G')].contact.mean(),'nanprob',sub.prob.isna().sum())
json.dump({'pair_names':names,'pair_weights':[float(x) for x in w],'wp':3,
           'ground':json.load(open('ground_best.json')),'joint_mcc':r[0],
           'joint_thr_pair':r[1],'joint_thr_ground':r[2],'rate_targets':{'pair':tgt_p,'ground':tgt_g},
           'test_pos_rate':float(sub.contact.mean())},open('blend_final3.json','w'),indent=2)
