# mirfak_iota — 🏁 FROZEN FINAL: `<metric>0.20576<\metric>` — submission validated

Refs: [[mirfak_iota_final_v2.md]] · [[castor_zeta_final_mshb_avg10.md]] · [[betelgeuse_mu_merge_v14_request.md]]

## Final state of `final_solution/`
- `submission.csv` — 25680 rows, `id_seqpos` order identical to `sample_submission.csv`, no NaN, values ≥ 0.
- `code/` — full reproducible pipeline + `README.md` + `models_registry_final.json` (16 models) + `ensemble_test_pred.npy` (exact `(240,107,5)` test prediction used).

**Final CV (canonical 5-fold, SN_filter==1, MCRMSE over 3 scored cols): `<metric>0.20576<\metric>`**
cols: reactivity **0.1890**, deg_Mg_pH10 **0.2341**, deg_Mg_50C **0.1942**.

## What is in the final blend
- `struct_scored` 11 seeds (0.20952), `struct_all5` 11 seeds (**0.20862**), `struct_huber` 2 seeds (0.21422) — my 3-channel base-pair-contact CNN+BiGRU.
- betelgeuse `mscnn448b10` (0.2225), `mscnn384b10` (0.2230), `cnn384b10` (0.2327), `lstm256b10` (0.2262), `mscnn512b10`.
- castor `mshb_avg10` (0.21597), `mshb_s45`, `lshb_avg2`.
- my `l512s42`; all-5 `be_mscnn384b` supplies unscored columns.
Post: ridge-NNLS per column → similarity-gated kNN (sim^4, k=5, thr .85, w .2, clip 0).

## Tried and rejected (CV-verified, so nobody repeats them)
struct LSTM bag (0.20577 — no gain), wide/deep struct (fold0 0.2228), 3-channel→scalar descriptor features, GBDT stacking, per-loop/per-position blend weights, Gaussian smoothing, reverse TTA, ridge/mean/greedy vs NNLS, per-regime gate, curated-vs-full registry averaging. Everything is documented in earlier messages.

## Final CV trajectory (my messages)
0.2676 → 0.2226 → 0.2163 → 0.2102 → 0.2071 → 0.2060 → **0.20576**

Freezing here (~30 min before work_end). No more writes to `final_solution`. The coordinator can grade. Good swarm run — the diversification (struct2 seeds + be CNN/LSTM + ca mshb/hb2) is what took us from 0.23 to 0.206.

#submission #status #ensembling #insight
