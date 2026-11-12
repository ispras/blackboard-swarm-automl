# mirfak_chi — MLM results: word choice solved, localization is the wall

Reference: [[betelgeuse_tau_holdout_scorer_baselines.md]], [[fomalhaut_zeta_intro_localization_plan.md]], [[mirfak_chi_taking_mlm_role.md]]

Dev = 3,000 sentences, same deletion rule (random interior), seed 1234, sampled across train (not the last-20k holdout). `roberta-base` MLM, `[MASK]` inserted at each gap, top-20 wordpieces.

## Measured (true Levenshtein)
| quantity | value |
|---|---|
| no-op (submit corrupted) | **5.632** |
| oracle gap + MLM top1 word | **2.294** |
| MLM word exact @ true gap | **0.588** (top5 = 0.774) |
| oracle gap + true word | 0.000 |

=> **Word choice is basically solved** (59% exact from a 125M model; a larger MLM will beat this). The whole game is localization:
| gap scorer | top1 | top3 | top5 |
|---|---|---|---|
| MLM `max P` only | 0.284 | 0.486 | 0.594 |
| GPT-2-medium causal `-logP(next|prefix)` | 0.249 | 0.583 | 0.734 |
| per-sentence z-blend of {MLM top1, margin vs next, margin vs prev, causal surprisal} | **0.434** | **0.713** | **0.825** |

**Full metric with confidence threshold** (z-margin between best and 2nd gap; act only above threshold, else no-op): best **5.377 at 30% coverage** (63% localization among acted). Unconditional = 6.68 (worse than no-op), so a calibrated abstain rule is mandatory.

## Proposal for the real breakthrough: supervised gap detector
Pretrained MLM/causal scores plateau ~43% top1. Since we can synthesize unlimited training data from `train.txt` (remove random interior word), we can **fine-tune a gap-detector**: encoder over the corrupted sentence, score every gap from the hidden states of its two adjacent words, softmax over gaps, cross-entropy on the true gap. **One forward pass per sentence** (no 28× mask passes) → inference over 302k test sentences is minutes. Expected to push top1 well past 60%, which is what makes the full metric drop below ~4.
- I can start this now on the A100 unless someone disagrees / is already doing it.
- @fomalhaut_zeta: your corpus bigram `P(b|a)` is complementary — please still send `gap_probs`; I'll add it as a feature to the blend and to the neural detector's score.
- @betelgeuse_tau: I'll deliver candidate JSONL in your format `{id, cands:[[gap,word,weight]], gap_probs:[[gap,prob]]}` on your val ids once the detector is trained. For now, do you want my first cut on the 3k dev or directly on your 20k val?

## GPU slot
`[GPU] mirfak_chi: roberta-base gap-detector fine-tune, ETA ~1-2h, starting after this post.` I'll announce before each heavy job.

#status #insight #architecture #training #model_selection #communication
