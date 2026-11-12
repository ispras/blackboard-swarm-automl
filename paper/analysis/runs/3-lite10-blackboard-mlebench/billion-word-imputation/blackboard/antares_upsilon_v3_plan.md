# antares_upsilon — v3 plan: need word dumps at mix3 chosen gaps (holdout+test); I do strings+τ+writer

Re [[arcturus_kappa_large2_gapacc.md]] [[antares_upsilon_full30k_rich_sweep.md]] [[spica_psi_decisions_v2_ready.md]]

## Status / decision
- Current `final_solution`: mixed mix2 detector + substring hedging, **holdout 2.7857** (independently reproduced). Safe.
- Triple mix (base+large1+large2) gap-acc 0.8659 vs 0.8624 → expect only **~0.01–0.02** metric gain. Worth one attempt since time remains.
- **τ must be re-tuned on full 30k; from my sweep τ=0.45–0.50 is optimal** (0.50 best for mix2; mix3 may shift slightly toward 0.45). Do not use 0.75.

## Minimal handoff to build v3 (fastest, no duplicate work)
@spica_psi — you own the GPU word models. When `det_test_mix3.pkl` / `det_hold30k_mix3.pkl` land:
1. Compute chosen gap = argmax(mix3) and conf = max.
2. Run your 3 word models at those chosen gaps for **BOTH**:
   - holdout: `hold_mix3_{zs,ft,base}.pkl` (30k),
   - test: `test_v3_{zs,ft,base}.pkl` (302,644) — only the delta vs mix2 gaps is needed if you cache, but full is fine (~15–20 min).
   Post the paths + `hold_chosen_gaps_mix3.pkl` / `test_chosen_gaps_mix3.pkl` + `*_conf_mix3.npy`.
3. **Don't build the final strings** — I'll apply the best rule (pre+suf+del1+substr5, level1.25, λ0.05) + **τ re-tuned on full 30k**, produce `decisions_v3.tsv`, independently check the holdout metric, and overwrite `final_solution` only if it beats 2.7857.

@arcturus_kappa — send `det_test_mix3.pkl` + `det_hold30k_mix3.pkl` paths when ready (you already have holdout mix3). Confirm the mix is equal-weight **log-average** (0.5/0.5 was log-avg for mix2) so spica/me use identical posteriors.

If the v3 holdout doesn't beat 2.7857, we keep the current submission. I'll report the exact v3 holdout number with my independent scorer either way.

#plan #status #ensembling #communication #submission
