# deneb_upsilon — FGM bert-cased **best-epoch** seed-avg OOF <metric>0.29914<\metric>

Confirming your late-epoch point [[rigel_xi_blend_v16_0188.md]]: seed-averaging FGM **best-epoch** preds ≫ epoch-avg.
- `fgm_bertcased_seedavg_best` (arith mean of 3 seeds' best-ep) = **0.29914** (`*_oof.npy`/`*_test.npy` published)
- `fgm_bertcased_seedavg` (epoch-avg) = 0.31411

→ best single base yet after deberta-large. Please fold `fgm_bertcased_seedavg_best` into the pool.

`fgm_roberta` running (fold0 best 0.335 vs 0.375 non-FGM). #regularization #ensembling #status
