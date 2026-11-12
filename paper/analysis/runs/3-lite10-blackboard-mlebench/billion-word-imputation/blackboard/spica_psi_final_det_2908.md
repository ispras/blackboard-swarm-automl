# spica_psi — FINAL detector ck15625 + word ensemble: <metric>2.9082</metric> on full 30k holdout

Re [[spica_psi_full30k_result.md]] [[antares_upsilon_final_plan_confirmed.md]] [[arcturus_kappa_status_gpu_full.md]]

## Full-30k holdout (exact char-Lev, baseline 5.5465), detector `ck15625` argmax gap + 3-way word ensemble (0.5/0.3/0.2, expected-Lev top-20, temp1.5, λ0.05)
| τ | metric | apply | loc-prec | word-acc |
|---|---|---|---|---|
| 0.55 | 2.9110 | 86.0% | 0.927 | 0.590 |
| **0.60** | **<metric>2.9082</metric>** | **83.7%** | 0.937 | 0.597 |
| 0.65 | 2.9203 | 81.7% | 0.945 | 0.603 |

Detector gap-acc by ckpt: ck4000 0.819 → ck11000 0.8417 → ck14000 0.8466 → **ck15625 0.8480**. So switching ck11000→ck15625 gains ~0.06 metric; **τ=0.60** is optimal (0.55–0.65 flat).

## Status of my deliverables
- `decisions_ck11k.tsv` + `submission_ck11k.csv` already built & validated from `det_test_ck11000` (302,644 rows, ids 0..302643, 259,292 inserts = 85.7% apply, every row verified: insertion or exact-unchanged). This is a **safe fallback**.
- arcturus is running `det_test_final` (ck15625) now. As soon as it lands I'll rebuild chosen gaps → word ensemble at those gaps → **`decisions_final.tsv`** with **τ=0.60**, ~20 min. ~7% of chosen gaps may shift vs ck11000; I'll rerun words fully to be safe.
- @antares: please write `final_solution/` from `decisions_ck11k.tsv` **now** as the safe baseline if you like, then replace with `decisions_final.tsv` when I post it (it should be ~0.05 better). I will not touch `final_solution/`.

#status #ensembling #model_selection #submission #communication #insight
