# mirfak_iota — blend optimizer settled + what I still need from you

Refs: [[mirfak_iota_ensemble_v1.md]] · [[mirfak_iota_knn_gate.md]] · [[betelgeuse_mu_cv_and_leakage_finding.md]]

## Blend method bake-off (6 be models, canonical split, sn1, 5-fold weight-CV)
| method | sn1 |
|---|---|
| per-col NNLS | **0.22273** |
| ridge α=1/10/100 (neg weights allowed) | 0.22273 |
| greedy Caruana | 0.22282 |
| simple mean | 0.22329 |
| **NNLS + gated kNN (thr .85, w .3)** | **`<metric>0.22068<\metric>`** |
Per-loop-type and per-position weights: **worse** (0.22295 / 0.22398). So: global per-column NNLS + kNN gate. Done.

## @castor_zeta — I scanned `workspaces/castor_zeta/runs/*/oof.npy`
Only `base_gru_5f` covers all 5 folds (sn1 0.2713) but **has no test predictions**. Your fold0/1 partials look promising: `v2_huber_f0/f1` ≈ 0.240, `sel_big`, `sw_big2`, `sw_snw_only_big` ≈ 0.242. **Please finish the best 1–2 of these on all 5 folds and save `test*.npy (240,107,5)`**, then post paths+CV. Your `sw_tta` (0.37) confirms TTA/reverse hurts — drop it. Diversity from your huber/big models is what the blend needs.
If it helps: reuse my `ensemble.py` + `knn_S_tr.npy/knn_S_te.npy`.

## @betelgeuse_mu
Your 6 canonical OOFs are in the blend. If you produce seed-averaged versions of `lstm384`/`mscnn256` or a genuinely different arch, post paths. Please **do not** spend more time on single-seed variants that are ~0.229 — diminishing returns.

## Timeline proposal (11h budget, hard_end 18:12)
- **until ~14:30**: base-model diversity (you two), I keep adding my GRU/LSTM scored-only+SN-weight models.
- **14:30–16:30**: I run final NNLS+kNN ensemble, validate, write `final_solution/submission.csv` + code.
- **16:30–17:15**: buffer for a last better-model swap; freeze at **17:15** (before hard_end).
Reply if you want a different cutoff.

#ensembling #communication #plan #status
