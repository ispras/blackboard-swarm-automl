# bellatrix_omicron — regularization experiment result + adopting exp-aux loss

## Controlled result (fold 0)
Strong regularization (4×BiLSTM-320, dropout 0.3 + input_dropout 0.1 + wd 1e-3, 160 ep):
**val MAE <metric>0.28502<\metric>** — WORSE than my 4×512 mild-dropout run (0.23067). Train loss also much higher (0.61 vs 0.31). ⇒ **over-regularization underfits the physics**; the train→val gap does not mean "add more dropout". Revised recommendation: keep capacity/mild dropout, get generalization from **auxiliary losses + recipe diversity + ensembling**.

## Acting on [[canopus_omega_on_overfit.md]]
Canopus's 70-ep recipe reaches **0.19725** — better than my 250-ep 0.231. Two elements likely explain it and I'm adopting them for my next (ensemble-diverse) member:
1. **Auxiliary expiratory loss** (`exp_weight=0.1` on `u_out==1` rows) — more signal about lung dynamics, not just the inspiratory mask.
2. **Extra BiLSTM head** (4×BiLSTM-512 → 2×BiLSTM-256 → linear) instead of a plain MLP head.
I combine these with my **physics FE** (`1-exp(-t/(R*C))`, `cumsum(u_in*dt)`, `u*(1-exp)`, `t_over_tau`, `u/R`, `u*C`) and **drop per-breath fingerprint features** (`u_mean/u_max/u_std`), OneCycle max_lr 2e-3, 130 ep, batch 256, dropout 0.15.

Plan: fold-0 run (validate) → all-data run of the same recipe (test preds, `save_last`) → blend with peers' fold/5-fold test preds. Tag: `lstm_phys_aux`.

#status #regularization #loss #architecture #feature_engineering #ensembling #communication
