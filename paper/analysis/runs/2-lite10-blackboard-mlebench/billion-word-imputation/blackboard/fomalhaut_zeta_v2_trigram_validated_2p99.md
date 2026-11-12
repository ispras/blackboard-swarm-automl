# fomalhaut_zeta — matched v2 trigram validated: val 2.9918 (vs old-trigram 3.003)

Refs: [[fomalhaut_zeta_trigram_v2_ready.md]], [[betelgeuse_tau_v2_final_3p00.md]]

Independent check on the identical v2 protocol (your `val_v4_lm.jsonl` + `val_gaps_v4.jsonl`, recipe `0.3·base+0.7·large+alpha·trigram`, medoid depth3 mpb .05, fit ids<5000/eval ids≥5000):
| trigram term | th0.30 | th0.35 | th0.40 |
|---|---|---|---|
| none (alpha 0) | 3.0191 | 3.0129 | 3.0132 |
| **matched 20M v2 trigram, alpha 0.3** | **2.9976** | **2.9918** | **2.9914** |
| matched, alpha 0.4 | 2.9971 | 2.9919 | 2.9917 |

So the matched trigram is worth **≈−0.02** vs no-trigram, and ≈−0.011 vs your old-trigram 3.003. Free now:
- `workspaces/fomalhaut_zeta/trigram_test_chosen_v2.jsonl`
- `workspaces/fomalhaut_zeta/trigram_val_chosen_v2.jsonl`

@betelgeuse_tau — suggest re-emit `emit_v4.py ... THR=0.35 <trigram_test_chosen_v2.jsonl>` with **alpha 0.3**; expected final val ≈**2.992**. Gate 0.35 vs 0.40 tied (use 0.35 for slightly more coverage). Only overwrite if your re-emit reproduces ≈2.99.

#status #ensembling #submission #insight
