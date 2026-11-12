import numpy as np
OUT='workspaces/sirius_tau'
v1o=np.load(f'{OUT}/cnn_ground_oof_all.npy'); v2o=np.load(f'{OUT}/cnn_ground_oof_all_v2.npy')
v1t=np.load(f'{OUT}/cnn_ground_test_all.npy'); v2t=np.load(f'{OUT}/cnn_ground_test_all_v2.npy')
assert v1o.shape==v2o.shape and v1t.shape==v2t.shape
np.save(f'{OUT}/cnn_ground_oof_all_avg.npy',(v1o+v2o)/2)
np.save(f'{OUT}/cnn_ground_test_all_avg.npy',(v1t+v2t)/2)
print('avg saved',v1o.shape,v1t.shape)
