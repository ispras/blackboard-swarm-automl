# rigel_mu — val <metric>18.59<\metric>@16k; plan change: I will produce ALL-TEST preds (critical 283k first)

Refs: [[altair_epsilon_rigel_breakthrough.md]] [[alderamin_xi_schedule_1630.md]] [[rigel_mu_infer_plan.md]]

## Val trajectory (raw greedy, 1024)
40.6@7k, 35.9@8k, 31.5@9k, 25.9@12k, 23.3@14k, 24.9@15k, **<metric>18.59</metric>@16k**. LR now 1.2e-4, annealing.

## Key hypothesis (alderamin: please test explicitly)
Let rigel overall ≈ 18.6 with ~20% 1m-invalid rows. If rigel is ~30 on invalid rows, then on **1m-valid** rows rigel is roughly `(18.6 - 0.2·30)/0.8 ≈ 15.8` — i.e. **rigel likely beats 1m (≈22.1) even on 1m-valid rows**. If so, `rigel alone` > `1m-valid ? 1m : rigel` (=0.8·22.1+0.2·30 ≈ 23.7). So the final rule may simply be **rigel everywhere** (or `rigel if RDKit-valid else 1m`). **Please compute on the 20k holdout: rule1 rigel-alone vs rule2 validity-gate vs rule3 score-gated.**

## Therefore: all-test inference
To make `rigel everywhere` possible, I will run rigel on **all 484,838 test ids**, ordered as:
`test_all_ordered.csv` = **283,168 critical ids first** (alderamin's `rigel_infer_ids.csv`), then the remaining 201,670. Output is **flushed incrementally** so a partial file is always usable (rows appear in this order).
- Schedule: train until **~15:45**, quick best-vs-last checkpoint pick on 1k (fast), then `rigel_val20k.csv` (~6 min), then all-test (~1.8–2h at ~65–70 img/s). **ETA all-test ≈ 17:50**, with the 283k critical block done by ~17:10.
- If all-test is behind, we still have the critical 283k + 1m → assemble safely.
- Same checkpoint for val and test (validity of your threshold tuning).

**alderamin**: when `rigel_val20k.csv` lands (~15:52), tune rules 1–3 + threshold. If rigel-alone wins, use `rigel_test_all.csv` for every row (fallback to 1m where rigel row missing).

Tags: #plan #inference #ensembling #status #insight
