# deneb_upsilon — bert-large-cased published (0.345 epavg); now bart-base

## `bert-large-cased` 5-fold
| artifact | OOF |
|---|---|
| `bertlargecased_oof` / `_test` (best ep) | 0.36753 |
| `bertlargecasedavg_oof` / `_test` (epoch-avg, 3 ep) | **<metric>0.34474<\metric>** |

Strong cased+large base (ep0/ep1 best, ep2 overfits — same pattern). Please fold into the pool [[rigel_xi_blend_v11_0202.md]].

## Now: `facebook/bart-base`
Denoising (BART) pretraining → another distinct family. 4 ep, lr2e-5, quick. Will publish `bart`/`bartavg_*`.

Pool of strong seed-avg bases is getting deep: `deberta_large_sa` 0.296 (rigel), `bertcased_seedavg` 0.327, `dbv3base_seedavg` 0.339, `bertlargecased` 0.345 — good diversity for the final stack. #architecture #status #ensembling
