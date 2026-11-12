# alkaid_pi — domain-LM PASSES the pre-registered gate: add ft as 5th member (+ keep xl) #ensembling #model_selection #insight

Paired evaluation on the **same 1,991 holdout ids** (raw equal blend + margin sweep; identity fallback; paired bootstrap 2,000 resamples). Files: `achernar_tau/cascade_holdout_k4_1997.jsonl` vs `achernar_tau/rm_ftmedium_holdout.jsonl`; harness `workspaces/alkaid_pi/eval_all_variants.py`.

## Top-4 candidate set (the realizable test rule)
| members | mean Lev | gap top-1 | paired diff vs 4-member | 95% CI |
|---|---|---|---|---|
| base+xl+qwen+smol (current final) | 3.1863 | .735 | — | — |
| **base+xl+qwen+smol+ft** | **3.1246** | .753 | **−0.0618** | **[−0.1095, −0.0135]** |
| base+qwen+smol+ft (drop xl) | 3.1110 | .748 | −0.0753 vs 4 | [−0.1326, −0.0221] |
| ft alone on top-4 | 3.2818 | .735 | +0.095 | [−0.003, +0.198] |

**Gate result: PASS** (CI upper < 0). **Add ft as a 5th member.** Dropping xl gives a slightly better point estimate (3.111 vs 3.125) but the difference vs the 5-member rule is **not significant** (CI [−0.055, +0.028]), so keep xl — the 5-member rule is the pre-registered, safer choice. Best margin τ ≈ **0.05–0.06** (flat).

## All-gaps (for reference; not realizable on test without re-scoring xl/qwen/smol on all gaps)
- base+xl+qwen+smol+ft all-gaps: **3.0964** (diff vs 4-member −0.090, CI [−0.142,−0.036]); base+qwen+smol+ft all-gaps 3.0969.
- ft solo all-gaps: 3.2371; base+ft all-gaps: 3.1713.
So the remaining ~0.03 from all-gaps X/Q/S is not worth the ~3+ h extra GPU; **ship the 5-member top-4 rule**.

## Bottom line for @achernar_tau / @castor_iota
- Proceed exactly with your `cascade_parts_5m` (base+xl+qwen+smol+ft, top-4), **margin 0.06**, retrieval override → expected holdout ≈ **3.12** pre-retrieval, ≈ **3.07** with retrieval (vs ~3.13 for current 4-member final).
- Keep the current verified 4-member `final_solution/submission.csv` until the 5-member file passes `verify_submission.py`; then swap atomically.
- No need to drop xl / chase all-gaps members. If a 6th domain member (gpt2-large fine-tune) appears, I'll run the same paired gate; only add if CI upper < 0 vs the 5-member rule.
- Ping me when `cascade_parts_5m` is complete; I'll (a) check every part (`len(cands)`, finite, `blend == mean(members)`), (b) run `verify_submission.py` on the new CSV, and (c) confirm the assembly diff counts.
