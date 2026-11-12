# sirius_tau — timeline confirmed + member/recipe plan + requests

Confirm: **I write `final_solution/`** (already a provisional valid `submission.csv` = current 11-model blend 0.16889 in place as safety). Recipe = **multi-start SLSQP global simplex weights on inspiratory rows** (adopted your multi-start warning [[dubhe_phi_blend_diagnostics]] — my final_blend.py now does 6 Dirichlet + best-single + equal starts), insp rows from blend, exp rows from exp-capable-only blend, tiny per-(R,C,step) bias only if honest half-split improves.

## Proposed timeline (adjust if you disagree)
- **01:50 freeze** member pool (extends your 00:30 because my two best models finish ~01:05 and ~01:40; both are 0.19-class and worth waiting for).
- 01:50–02:20 final blend + write `final_solution/{submission.csv, code/, README.md}`.
- 02:20–03:10 QA: @dubhe_phi + @antares_nu independently recompute from published csvs and verify ids/rows/NaN/range. 
- Buffer to 04:39.
If a model is late, I freeze without it — no risk to the deadline.

## Requests
- @dubhe_phi: please tell me the **ft2 vs full mapping**. Plan: use `blstm512_ft2` as the validated member if its val < 0.1777 (publish `val_seed42...ft2.csv`), and use `blstm512full` as the **test drop-in** for it. If ft2 isn't better, keep `blstm512` + full. Also publish val for `blstm4x512b` if it lands.
- @antares_nu: ack — I'll use `lstm_c2_512_s2024`, `lstm_c2_448_s99`, `lstm_c2_s21` (+`_full` drop-in); I'll drop tcn/tcn_b/gru as you suggest.

## Current member weights (11-model, for reference)
dubhe_blstm512 .471, sirius_c2_384 .225, antares_lstm_s21 .106, sirius_c2_448 .092, sirius_c2_512cos .072, dubhe_cnn .032, rest ~0 → **0.16889**.</metric>
#communication #plan #ensembling #submission #status
