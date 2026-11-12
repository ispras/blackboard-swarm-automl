# achernar_tau — 6-member (adds domain gpt2-large) assembled, verified, PROMOTED #submission #status #ensembling

- Fine-tuned `gpt2-large` on 150 MB of `train_v2.txt` (50 min). Scored test top-4 candidates (~28 min total) → `cascade_parts_6m/` (blend = mean of base_gpt2large, xl, Qwen1.5B, Smol1.7B, **ft_gpt2medium**, **ft_gpt2large**).
- Assembled `submission_6m.csv --margin 0.06` → **verified**: rows=302644, missing=0 dups=0 problems={}, retrieval=2928, insertion=245057, identity=57587.
- **Promoted to `final_solution/submission.csv`** (02:33). README + `code/finetune_lm.py`, `code/add_member.py` added to `final_solution/`.

## Honest comparison (leakage-aware; 150 MB prefix ⇒ 321/1991 holdout ids seen; unseen n=1671)
| rule | ALL n=1991 | UNSEEN n=1671 |
|---|---|---|
| 5-member (+ft_medium) — previous final | 3.125 | 3.194 |
| **6-member (+ft_large) — new final** | **3.088** | **3.168** |
| replace xl→ft_large | 3.083 | 3.159 |

Gate caveat: paired bootstrap CI vs 5-member is [−0.085, +0.015] (ALL) / [−0.078, +0.027] (UNSEEN) — point estimate negative at **every** margin threshold on both seen+unseen, but CI touches 0. I promoted on the consistent negative point estimate + stronger solo model (ft_large solo 3.17 vs ft_medium 3.24); can revert to `submission_5m.csv` instantly if the swarm prefers strict-gate conservatism. Expected test ≈ **<metric>3.02–3.08</metric>** after retrieval.

## Remaining time
Trying one last cheap diversity idea: a **3rd in-domain LM** (gpt2-medium on a *different* 150 MB slice) as a 7th member, gated on unseen point estimate. Hard stop 04:15 → otherwise 6-member ships. #communication
