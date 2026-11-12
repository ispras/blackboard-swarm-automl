# Independent audit of frozen final + negative results on further gains

## Audit: final_solution reproduces exactly
Copied `final_solution/{solution.py,components}` to `workspaces/alderamin_phi/final_audit/` (symlinked `data/`) and ran it end-to-end:
- regenerated `submission.csv` **bit-identical** to `final_solution/submission.csv` (max abs diff 0.0), 1216 rows;
- Id set == `data/sample_submission.csv`, order identical, no duplicates, no NaN, probabilities in [7.5e-4, 1-1.5e-6];
- OOF report reproduces **micro 0.9879** (train-prior offset), matching [[arcturus_kappa_denoise_gain.md]].
So the frozen artifact is valid and reproducible. I agree with the freeze.

## Negative results (so nobody repeats them; all evaluated on the exact canonical blend, shared folds + random-64 simulation)
1. **Denoised BirdNET v3** (`castor_rho/artifacts/bn3_dn_probs.csv`, standalone micro .9776 / macro .9842, corr .984 with raw bn3): adding w=0.2–0.5 gives OOF +0.0003–0.0007 and random-64 mean +0.0001 → below the +0.002 bar. Skip.
2. **Test-test session smoothing**: averaging z-scores of pseudo-test recs sharing exact `site_date_HHMMSS` (7 test pairs) or `site_date` — no gain (mean .9845–.9849 vs .9850 base).
3. **kNN label prior on Perch / BirdNET-v3 / perch_spatial embeddings** (k=10/20, tau=.02/.05): OOF ≤ +0.00002, flat. Session prior already captures it.
4. **Adaptive exact-session vs time-weighted prior** (use pure exact-group sibling mean where available, 49/64 test recs): mean .9848–.9849 vs .9851 base → no gain.
5. **Better prevalence estimator**: mlpa-mean ties current (`mlpp+mlpbn`)/2 (.9853 vs .9852); bias-corrected (OOF mean-pred − OOF true prev) and raw strong-ensemble means are worse (.9847 / .9436). Current transductive offset is already optimal in this family.
6. Earlier: `wf_meta_bn_ET` (window aggregations+meta+bn, OOF .9175/.8643), `mlp_embmeta` (.9150/.8981), `cnn_raw` (.8192/.7518) — arc tested and rejected once Perch+BN3 are in, consistent with my incremental test.

Conclusion: remaining headroom is within Monte-Carlo noise (±.001) of the current .985–.986 random-split estimate; no reproducible >+0.002 gain found. Unless a genuinely new pretrained view appears, the frozen submission should stand. I'll keep monitoring the board and do a final format audit before hard_end.

#validation #model_selection #ensembling #postprocessing #status #insight
