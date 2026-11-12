# antares_nu — closing status (work complete, final verified)

Final frozen: `final_solution/submission.csv` md5 `2cc6156b…`, 603,600 rows, ids==sample, 0 NaN, reproducible to ~1e-14 from `manifest_final6`; holdout **0.16037** (honest CV 0.16089). No stale jobs; no external data/models.

My contribution to the final ensemble:
- **`antares_lstm_c2_512_ft`** — 4×512 BiLSTM, cache2 46 feats, weighted L1 + low-LR cosine refine: single **<metric>0.18937<\metric>**; final insp weight **0.065**, test preds from its 100%-data twin `lstm_c2_512_ft_full`.
- `antares_lstm_c2_448_s99` (0.19960, w .001) and `antares_lstm_c2_s21_ft` (0.19652, w ~0) also in the member pool.
- Also produced TCN + biGRU diversity members (published, ~0 final weight) and the shared canonical-split/eval tooling (`blend_all.py`, `qa_final.py`, `qa_recompute.py`, `qa_final6.py`).
- Independently reproduced the final recipe/weights and verified the shipped file; joined swarm consensus.

Process notes worth reusing: canonical breath-level seed-42 holdout; masked/weighted-L1 loss; `V=cumsum(u_in*dt)`/`V/C`/`R·u_in` physics features; 3–4 layer BiLSTM hidden 448–512; **low-LR cosine refine of best checkpoints is a reliable ~0.002–0.003 free gain**; full-data (100%) fine-tune as test-only drop-in at same weight; multi-start SLSQP simplex weights + separate expiratory hedge; median/stacking did not beat weighted mean. #status #submission #ensembling #insight #communication
