# spica_psi — FINAL decisions ready (ck15625 det, τ=0.60, 3-way words) → @antares please write final_solution

Re [[spica_psi_final_det_2908.md]] [[antares_upsilon_final_plan_confirmed.md]]

## Deliverable (validated)
- **`workspaces/spica_psi/decisions_final.tsv`** — `id<TAB>gap_k<TAB>word`, **253,439 rows** (apply 83.74%), ids unique in 0..302643.
- Optional full file: `workspaces/spica_psi/submission_final.csv` (302,644 rows).
- Built from: final detector **`det_test_ck15625`** argmax gap (convention k = argmax_index+1) + word ensemble `roberta-large 0.5 ⊕ rlarge_ft 0.3 ⊕ base ckpt4000 0.2`, expected-Lev top-20 (temp1.5, λ0.05), **τ=0.60**. 6.0% of chosen gaps changed vs ck11000; I re-ran those 18,184 (delta) so all words match the final gap.
- Independent structural validation: 302,644 unique ids in order, every inserted row = exact token-insertion at k with a whitespace-free word, every non-inserted row byte-identical to test sentence, gap bounds `1≤k≤M-1` hold, header `id,sentence`, inner `"` doubled. **0 bad rows.**

## Expected score
Full-30k holdout with this exact recipe: **<metric>2.9082</metric>** (τ=0.60: apply 83.7%, loc-prec 0.937, word-acc 0.597). Test apply rate 83.74% matches holdout → detector calibration is consistent. Do **not** use ck11000 version if you already wrote it; replace with `decisions_final.tsv`.

## Why this gap choice is final
Detector gap-acc: ck4000 .819 → ck11000 .842 → ck14000 .847 → **ck15625 .848**. Checkpoint averaging (14k+15625, 11k+14k+15625) did **not** help (0.8477/0.8474 < 0.848); adding base-MLM `maxp` to gap choice gave only .8485. So **det-only argmax + τ=0.60** is the right, simplest final rule.

Still open (~7h left): I'm testing an extra word model (different model family) to add to the word ensemble; if it beats the current 1.944@oracle word cost I'll send a `decisions_final_v2.tsv`. Otherwise `decisions_final.tsv` stands.

#submission #status #ensembling #model_selection #communication
