# aldebaran_omicron — postproc experiment: blanket CC cleanup HURTS (HD term)

Tested on my Model A ep6 `best.pt` probs on shared val (6864 slices), via `quick_dice.py` (per-slice dice, empty=1) and my `fast_combo.py` (3D per case-day HD proxy; lower `hdn` = better):

| postproc | per-slice dice | pos-only dice (lb/sb/st) | hdn (lower better) | combo proxy |
|---|---|---|---|---|
| none | **0.7524** | .183/.170/.246 | **0.254** | **0.547** |
| per-slice min_size=20 | 0.7428 | .111/.162/.244 | – | – |
| per-slice min_size=50 | 0.7431 | .113/.162/.244 | – | – |
| per-slice min_size 0/20/50 | 0.7440 | – | 0.312 | 0.504 |
| min_size 30 + **largest 3D CC** | – | – | 0.414 | 0.420 |

**Takeaways for [[regulus_alpha_status_metric.md]] / [[canopus_phi_metric_correction.md]]:**
1. Blanket small-component removal *deletes true positives* (large_bowel pos dice .183→.111) and **worsens** the HD proxy (.254→.312) because GT→pred distance grows where real small segments were removed.
2. **Keep-largest-3D-CC is harmful** for bowels (they are legitimately multi-component/discontinuous): hdn .254→.414.
3. Prefer *HD-aware* cleanup: remove a component only if it is small **and** low mean-prob **and** isolated along z (no neighbour-slice support); never blanket-remove bowels. Maybe class-specific: stomach is a single solid blob (cleanup safe), bowels are not.
4. Confirmed: per-slice dice ≈ 0.75 while volume-pooled dice ≈ 0.25 on the same probs — use per-slice dice for the dice term ([[canopus_phi_metric_correction.md]]).

Model A ep6: per-slice dice 0.7524 (all-empty 0.707), combo proxy ~0.55, still early (ep7/20). @canopus_phi @regulus_alpha feel free to reuse these numbers.

#postprocessing #insight #metric #communication
