import pandas as pd, numpy as np
OUT='workspaces/sirius_tau'
gi=pd.read_parquet(f'{OUT}/ground_all_index.parquet')
oof=np.load(f'{OUT}/cnn_ground_oof_all.npy')
tr=pd.DataFrame({'contact_id':gi.game_play+'_'+gi.step.astype(str)+'_'+gi.nfl_player_id_1.astype(str)+'_G',
                 'ground_cnn_prob':oof})
tr.to_parquet(f'{OUT}/ground_cnn_feat_train.parquet',index=False)
ti=pd.read_parquet(f'{OUT}/ground_test_index.parquet')
tp=np.load(f'{OUT}/cnn_ground_test_all.npy')
te=pd.DataFrame({'contact_id':ti.game_play+'_'+ti.step.astype(str)+'_'+ti.nfl_player_id_1.astype(str)+'_G',
                 'ground_cnn_prob':tp})
te.to_parquet(f'{OUT}/ground_cnn_feat_test.parquet',index=False)
print('train',tr.shape,'test',te.shape)
print(tr.head())
