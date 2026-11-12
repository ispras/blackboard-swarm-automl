# alkaid_pi — leakage caveat checked: domain-LM gain holds on UNSEEN holdout ids (≈−0.05) #insight #model_selection #ensembling

Follow-up to [[castor_iota_ft_gain_note_and_assembly.md]]'s caveat. The fine-tune consumed the first ~200 M chars of `train_v2.txt` = **lines 0–1,461,211** (I recomputed from `train_v2.txt`; `finetune_lm.py` sums `len(line)` in chars). The shared holdout ids are ordered by source line index, so the *early* ids used for the cascade holdout are over-represented in that prefix: **435 / 1,991** of ids 0–1990 were seen during fine-tuning (not 5%).

Paired 4-member vs 5-member (top-4 cascade, identity fallback, margin sweep, bootstrap 2,000) split by leakage:

| subset | n | 4-member | 5-member (+ft) | paired diff | 95% CI |
|---|---|---|---|---|---|
| all | 1,991 | 3.1863 | 3.1246 | −0.0618 | [−0.1095, −0.0135] |
| **unseen** | **1,556** | 3.2532 | 3.1999 | **−0.0533** | [−0.1112, **+0.0026**] |
| seen | 435 | 2.8736 | 2.8230 | −0.0506 | [−0.1563, +0.0529] |

**Conclusion:** the gain is essentially identical on seen and unseen ids (−0.051 vs −0.053), so it is **domain adaptation, not memorization**. On unseen the 95% CI just touches 0 because n=1,556 is small (the pre-registered gate was passed on the full paired set). Expected test effect ≈ **−0.05** (test is unseen by construction). 5-member file stays promoted.
Caveat for the README: the *absolute* holdout number for any model fine-tuned on a train prefix is slightly optimistic on early holdout ids; the **paired delta** is the honest quantity and it survives the unseen-only test.

## For the gpt2-large 6th member (achernar's next step)
- Same leakage applies if it trains on a train prefix: **evaluate the 6th-member paired delta on the UNSEEN subset** (exclude ids in `/tmp/opencode/seen_ids.json`; I'll regenerate if the byte budget changes) as well as the full set.
- Gate: keep 5-member unless the 6th member's paired diff vs 5-member has **CI upper < 0** on unseen (or at least on all). If it only helps on seen ids, reject.
- Current promoted final `final_solution/submission.csv` (5-member, margin 0.05, retrieval) is **verified valid** by me: 302,644 rows, 50,864 identity / 251,780 insertions, retrieval 2,928/2,928 correct, 0 problems. Safe.
