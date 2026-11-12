# achernar_tau — domain gpt2-large: helps as 6th member (unseen −0.026); scoring test now #insight #ensembling #status

Fine-tuned `gpt2-large` on **150 MB** of `train_v2.txt` (1 epoch, ~50 min). Scored all-gaps holdout → `workspaces/achernar_tau/rm_ftlarge_holdout.jsonl` (schema `{id,cands:[[gap,word,mean,total]]}`).

Honest paired numbers (leakage-aware; 150 MB prefix ⇒ **321 / 1,991** holdout ids seen; unseen n=1,671):
| rule (top-4 by base, raw blend + margin) | ALL n=1991 | UNSEEN n=1671 |
|---|---|---|
| 5-member base+xl+qwen+smol+ft_medium (current final) | 3.125 | 3.194 |
| **6-member + ft_large** | **3.088** | **3.168** |
| replace xl → ft_large (base+ft_large+qwen+smol+ft_medium) | 3.083 | **3.159** |
| replace ft_medium → ft_large | 3.124 | 3.202 |

- ft_large **solo** all-gaps = 3.174 (vs ft_medium 3.237) → bigger domain model is stronger, as expected.
- The gain survives on **unseen** ids (−0.026 for 6-member, −0.035 for replace-xl), so it's domain adaptation, not memorization.

## Decision / plan
Safer pre-registered choice = **add ft_large as 6th member (keep xl)**; the replace-xl variant is ~0.01 better on unseen but that's within noise, and dropping xl was previously non-significant, so I'll default to **6-member**. Now scoring the test top-4 candidates with ft_large (~1 h, `cascade_parts_6m/`), then assembling `submission_6m.csv --margin 0.06`, verifying, and promoting **only if** it verifies and beats 5-member on the paired gate.

@alkaid_pi: please run the pre-registered gate on unseen (rm_ftlarge vs cascade_holdout_k4) — and, if quick, also the replace-xl variant (I'll produce its dump if it matters). Hard stop 03:30: if not promoted by then, 5-member stays. #communication #submission
