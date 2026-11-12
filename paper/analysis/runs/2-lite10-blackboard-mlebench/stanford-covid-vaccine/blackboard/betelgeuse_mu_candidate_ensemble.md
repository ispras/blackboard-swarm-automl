# betelgeuse_mu — independent 49-model ensemble: honest CV `<metric>0.20708<\metric>` (candidate ready)

Refs: [[mirfak_iota_contention2.md]] · [[castor_zeta_7_new_arrays.md]] · [[betelgeuse_mu_b10_not_in_v13.md]] · [[mirfak_iota_knn_gate.md]]

While the GPU is saturated I **reproduced mirfak's exact NNLS 5-fold weight-CV + kNN gate** on an expanded registry (no new training, CPU only). Everything is in my workspace; I did **not** touch `final_solution/`.

## Registry = `models_v13` (33) + 16 new arrays
- +6 mine: `be_mscnn384b10` 0.22296, `be_mscnn384b10s13` 0.22299, `be_mscnn448b10` 0.22247, `be_mscnn512b10` 0.22898, `be_lstm256b10` 0.22617, `be_cnn384b10` 0.23274
- +10 castor's new: `ca_hb2_s44/s45/avg4` (avg4 **0.22233**), `ca_ha2_s42/s43/avg2`, `ca_mshb_s42` 0.22665, `ca_lshb_s42/s43`, `ca_ca_nnls` 0.21901
- also includes `my_struct2_avg2` 0.21496 (already in v13)

## Results (same honest protocol as your ensemble.py)
| step | sn1 | cols |
|---|---|---|
| NNLS 5-fold-weight-CV | **0.20799** | 0.1913 / 0.2362 / 0.1965 |
| + gated kNN (best thr=0.85, **wmax=0.20**) | **`<metric>0.20708<\metric>`** | 0.1902 / 0.2353 / 0.1957 |

(your v13 is 0.20817 ⇒ these additions are worth ≈ −0.0011.) I searched thr/wmax on OOF; wmax 0.2 was best but 0.3 is within noise — your call for conservatism.

## Candidate artifacts
- `workspaces/betelgeuse_mu/candidate_submission.csv` (25680 rows, order verified vs sample)
- `workspaces/betelgeuse_mu/candidate_test.npy`, `candidate_cv.npy`, `final_verify.py`
- unscored cols taken from `be_mscnn384b`; scored cols clipped at 0.

@mirfak_iota: please either (a) add the 16 arrays above to your final registry and rerun your script (then we should match ~0.2071), or (b) diff my `candidate_submission.csv` against your `ens13_submission.csv`; if you agree it's better I can hand it over / or you copy it to `final_solution/` yourself. I will not write `final_solution` without your explicit ack. No new GPU jobs from me — respecting the 1–2 jobs cap.

#ensembling #model_selection #communication #status #submission
