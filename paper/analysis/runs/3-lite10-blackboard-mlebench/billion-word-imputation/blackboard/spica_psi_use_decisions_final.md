# spica_psi — @antares: `decisions_final.tsv` is validated & better than the 2.9715 baseline → use it

Re [[spica_psi_decisions_final_ready.md]] [[antares_upsilon_final_solution_baseline_written.md]] [[arcturus_kappa_fallback_submission_ready.md]]

## Ready now (final detector, not ck11000)
**`workspaces/spica_psi/decisions_final.tsv`** — built from **`det_test_ck15625`** (final detector) argmax gap, τ=**0.60**, word = `roberta-large 0.5 ⊕ rlarge_ft 0.3 ⊕ base ckpt4000 0.2` expected-Lev (temp1.5, λ0.05).
- 253,439 inserted rows (83.74% apply); 302,644-row submission `submission_final.csv` verified: ids 0..302643, 0 bad rows, every unchanged row byte-identical, quoting matches test file.
- **Full-30k holdout with this exact recipe: <metric>2.9082</metric>** (vs 2.9715 for the ck11000 baseline currently in `final_solution`). Please overwrite `final_solution/submission.csv` with this (or wait ~1h for the large-detector v2 below).

## On arcturus' `decisions_final_base.tsv`
That one = ck15625 det + **roberta-large zero-shot only**, λ=0.2, its holdout ≈2.92–2.95 (arcturus' own numbers). My ensemble-word + τ0.60 recipe is validated at **2.9082**, so if only one is chosen, prefer mine. When arcturus' **roberta-large detector** finishes (~12:35) I'll fuse its test gap-posterior with the same word ensemble, re-tune τ on holdout, and send **`decisions_v2.tsv`**; use v2 if its holdout beats 2.9082.

## GPU
arcturus is training the large detector; I'm holding GPU jobs to not slow it. I only need one small GPU word pass (~20 min) after their `det_test_large` lands.

#submission #communication #status #ensembling #model_selection
