# castor_eta — final submission provenance (md5 + exact array list)

`final_solution/submission.csv` md5 **192ee70c4770b493de15f3d10ab1e1d3** (603,600 rows + header, written 02:46, sorted-breath → joined to sample `id`, clipped [-2,65], 6 dp).

Exact composition (weights sum 1.0; all arrays sorted-breath layout):
| w | array | md5 |
|---|---|---|
| 0.0430 | castor `bilstm_v1_all_test.npy` (all-data 3×384) | 47f6503fe7b2bd66caea67781d36c663 |
| 0.0650 | castor `bilstm_c2_all_test.npy` | 2ede52f13751532067f7b4554830d03a |
| 0.0650 | castor `bilstm_c2_all2_test.npy` | 35d3d8e07fa7e1fc75e3bd7d0cf0aab6 |
| 0.1275 | castor `bilstm_c2_250_test.npy` | 0267f2a856ef1f1c43232e0e34bf8e4f |
| 0.1275 | castor `bilstm_c2_all250_test.npy` | 3c788734cfd368578873984485789e70 |
| 0.1300 | castor `bilstm_e3_220_test.npy` (RC-EMA) | 6f4ecc56da9c1c63293bb67ce126896f |
| 0.1300 | castor `bilstm_e3_all_test.npy` | 6609983b81b1f1d084a254e8d16fdd8d |
| 0.0508 | canopus `test_pred_lstm384_5f2.npy` (5-fold avg) | 0c0c2793d9ec29eed319a506ddd0ae44 |
| 0.0102 | canopus `test_pred_lstm384_5f2_all.npy` | ad5be88b45023d34f58c28033e891cfc |
| 0.0110 | canopus `test_pred_lstm512_c2b.npy` | 85f8174265c06de6980ab85bdf0cb373 |
| 0.0720 | canopus `runs/c2s123_test.npy` | c4f3699a939fd8bb86da76fda492f960 |
| 0.0845 | canopus `runs/e3_s456_test.npy` (RC-EMA seed) | c64b1582ea391d5d466bf6d5feb363b3 |
| 0.0845 | canopus `runs/e3_s456_all_test.npy` | 77eb4961769f61445709b86412a67f31 |

Fold-0 validation of the corresponding weight vector (all at 80%-data versions): **insp 0.16033**, breath-half cross-fit ≈0.1605. References: [[castor_eta_new_best_blend_01612]], [[castor_eta_final_01603]], [[canopus_iota_e3seed_01606]], [[canopus_iota_blend_0161]], [[bellatrix_iota_refresh3]]. Full README + code in `final_solution/`. #submission #ensembling #communication
