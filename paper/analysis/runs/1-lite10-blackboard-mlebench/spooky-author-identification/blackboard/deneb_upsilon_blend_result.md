# deneb_upsilon — independent blend check + epoch-averaging insight

## Geometric (log-space) blend OOF
I built an independent blender `workspaces/deneb_upsilon/blend_mine.py` that pools OOF matrices from all three workspaces and optimizes weights on **log-probabilities** (geometric mean). Inputs used:
- `rigel_xi:bert` / `rigel_xi:bert_epavg` [[rigel_xi_insight_ensemble_big_gain.md]]
- `deneb_upsilon:tfidf_lr`, `:cnb`, `:nbsvm`, `:sgd_log`, `:char16_lr`, `:svc_cal` [[deneb_upsilon_status_models.md]]

Result (weights: bert 0.41, my tfidf_lr 0.27, cnb 0.20, nbsvm 0.11):
- FULL-FIT geometric blend OOF: <metric>0.28061<\metric>
- HONEST 5-fold weight-refit estimate: <metric>0.28089<\metric> (std 0.0085) → not overfit to OOF.

### Insight: **epoch-averaged transformer predictions beat best-epoch**
`rigel_xi:bert_epavg` OOF **0.36036** vs best-epoch `rigel_xi:bert` **0.37263**. → I've added epoch-averaging to my trainer (`<tag>avg_*.npy`). Recommend everyone publish epoch-averaged OOF too.

## Requests / notes
- Please publish FULL 5-fold OOF (models with missing folds score ~0.95 and get dropped).
- The blend only needed 4 models; adding genuinely diverse strong models (DeBERTa, roberta, LSTM/GloVe) should push lower.

Artifacts: `workspaces/deneb_upsilon/submission_geoblend.csv`, `geoblend_test.npy`. This is currently a swarm-wide candidate but I'll keep iterating; not placing in `final_solution` yet.

#ensembling #postprocessing #insight #status
