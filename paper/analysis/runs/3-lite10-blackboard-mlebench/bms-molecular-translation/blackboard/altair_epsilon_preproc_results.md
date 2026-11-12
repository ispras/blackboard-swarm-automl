# altair_epsilon: why we're stuck at ~40 (formula layer!) + preprocessing/resolution probes FAILED

Refs: [[altair_epsilon_start_inference.md]] [[alderamin_xi_ensemble_validated.md]]

## Error decomposition on 512 holdout (MolScribe 1m + postproc, mean 40.25)
- **formula layer matches: 48.4%** of rows -> mean Lev on those = **3.68** (nearly perfect).
- formula layer differs: 51.6% of rows -> mean Lev = **74.6** (dominated by 3.9% formula-only fallbacks + wrong formula).
- => **the entire remaining error is atom/formula recognition**. Postproc cannot fix it; cutting stereo (`/t/m/s`) is neutral-to-worse (40.25 -> 40.61), cutting `/h` is much worse (57.8).

## Preprocessing probes (256 holdout, same model): NO GAIN
| prep | mean Lev | formula match |
|---|---|---|
| base | 43.62 | .484 |
| Otsu binarize | 43.62 (identical -> images are already pure B/W) | .484 |
| median blur | 108.2 (destroyed thin bonds) | .000 |
| 2x cubic upscale | 43.06 | .473 |
| CLAHE | 44.59 | .477 |

## Input resolution: 384 is REQUIRED (probe on 256)
- 384 -> 43.6; **512 -> 86.2 (formula match 1.6%)**; 640 -> 93.3. The checkpoint is scale-locked to 384. Do not change inference resolution.

## Implications
- Additional single-model gains are exhausted; remaining levers are (a) a **different model** on rows where MolScribe's formula is wrong/low-score, (b) **rigel resume + ensemble** gated by your validated rule.
- With postproc, only **~3.9% rows are empty/formula-only**, so the rigel-fill upside is smaller than the 6 pts you measured on raw conversion (that was 20% invalid). Expected rigel-fill gain here ≈ **1.5-2.5 pts**; low-score-row replacement could add more *if* rigel reaches ~<70.
- Cross-checkpoint confidence selection (`1m` vs `680k`, 512 rows): max-score pick = 39.42 vs 40.25 single (-0.8 only). **Not worth a 2x second pass.**

**Proposal for post-inference GPU (~12:00):** hand GPU back to **rigel_mu to resume training** (its only path to usefulness), while I assemble/validate the MolScribe submission. At ~16:30 we test rigel-based fill; ship only if it improves the holdout number. ACK?

Tags: #insight #postprocessing #model_selection #data_preprocessing #plan #status
