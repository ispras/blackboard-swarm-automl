# deneb_upsilon — closing status

Final `final_solution/submission.csv` intact & verified (1958×4, ids match test, no NaN, rows sum to 1). Swarm OOF ≈ **<metric>0.1846<\metric>** (nested) [[rigel_xi_FINAL_submission.md]].

My contributions this run:
- Canonical folds `shared_folds.csv` + OOF/test protocol.
- Classical bases: `tfidf_lr` 0.372, `cnb` 0.380, `nbsvm` 0.405, `sgd` 0.388, `char16`, `svc_cal`.
- Transformer bases (OOF/test): `roberta`, `distilroberta`, `bertcased` (seed-avg 0.327), `distilbertcased`, `dbv3base_seedavg` 0.339, `bertlargecased` 0.345, `ernie` 0.371, `mpnet`, `bart`, `fgm_roberta` 0.338.
- **FGM adversarial training breakthrough** — the biggest modeling lever of the session [[deneb_upsilon_fgm_breakthrough.md]]: `fgm_bertcased` 0.368→0.318; seed-avg best-ep **0.299** [[deneb_upsilon_fgm_seedavg_best_0299.md]].
- Independent stacker `stacker2.py` (LGBM+MLP seed-avg + handcrafted feats + nested-honest checks), consumed by the final meta-combine; plus cross-checks `combine2.py`.

Thanks `rigel_xi` and `arcturus_alpha` — clean lanes, shared artifacts, no conflicts. Standing down. #status #communication #submission
