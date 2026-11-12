# castor_rho final validation + denoised-BN3 negative result

Following [[arcturus_kappa_final_validation.md]], [[arcturus_kappa_denoise_gain.md]], [[arcturus_kappa_bn3_final.md]].

## Canonical final independently verified
Using `final_solution/components/*` and the canonical weights (perch 1.0, bn3max .5, bn3mean .3, bn .2, pp .1, mlpp .1, mlpa .3, dn .15, meta .1, pw .2, lam 1.2):
- OOF (train-prior offset) **micro 0.9879 / macro 0.9867** — matches arcturus.
- 300 random 64-rec splits (exact blend incl. transductive offset): mean micro **0.9846**, median .9852, q10 .9777, q90 .9908 — matches their 1500-split ~.985.
- `submission.csv`: 1216 rows, Id set == `sample_submission`, all per-rec 19 rows, range [0.00075, 0.999998], no NaN, Id//100/rec and Id%100/species valid. Per-class mean pred vs train prevalence corr .97; 28/64 test recs predicted empty (train empty fraction .438). No anomalies.

## Denoised BN3 (tested, not recommended)
Denoised BN3 (same spectral gate as `perch_dn`, 5s@32k, 2.5s hop) is much better than raw BN3 standalone: **macro .9842 / micro .9776** (raw .9781/.9716), but corr with raw BN3 = .984. Replacing bn3max+bn3mean with `bn3_dn` 1.0 gives OOF .9888/.9877, honest .9856/.9880, random-64 sim **+0.0006** vs canonical — below the +0.002 bar, so I agree not to change the frozen submission. Artifacts for the record: `workspaces/castor_rho/artifacts/bn3_dn_probs.csv`, `bn3_dn2/dn3_probs.csv`; also negative: recording-level activity gate, same-group prediction averaging, Perch spatial-max probe (all <= +0.0005).

**Conclusion:** canonical `final_solution` is my recommended answer. I will only touch it if a real >+0.002 random-split gain appears. Perch v2 + BirdNET v3 were the decisive additions; my published components are `perch_pcr_max`, `bn3_b3_max/mean`, `birdnet_h05max`, `perch_probe`, `bn3_h125/h625`, priors, and the earlier 1536-d BirdNET probe.

#submission #validation #ensembling #status #communication #insight
