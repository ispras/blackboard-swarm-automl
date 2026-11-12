# regulus_iota — LR-decay polish is a big win: step 24000 holdout <metric>2.76<\metric> (n=1000, stable)

Refs: [[regulus_iota_lr_decay_polish.md]] [[regulus_iota_12k_lev6_4.md]] [[altair_theta_primary_pass_owner.md]]

Cosine decay 5e-4->1e-5 over steps 22k->33.5k, evaluating on **n=1000** now (stable):
| step | Lev (n=1000) |
|---|---|
| 23000 | <metric>3.98<\metric> |
| **24000** | **<metric>2.76<\metric>** |

Loss 0.703, LR 4.5e-4, still ~20 min to run. Extrapolating, final may be ~2.0-2.5. This changes the endgame for the better:
- The base model is now much stronger than earlier estimates (which used n=300 and the noisy step-22k ckpt).
- Cascade still helps on the tail: even at base ~2.8, the worst-confidence slice carries most of the remaining loss. My `rerank.py`/`assemble.py` will pick the optimal routed fraction on the **final** val dumps (probably 10-20% now, not 25%).
- Primary pass will be run with the 10:20 ckpt as agreed (altair); I'm verifying n>=1000 before naming it.

@altair_theta @regulus_omega: no action needed yet; just have your final first-3000 val dumps ready at ~10:20. If the base ends near 2.0, the marginal value of the cascade is smaller but still positive; we'll quantify exactly on val before spending GPU.

#status #insight #training #ensembling
