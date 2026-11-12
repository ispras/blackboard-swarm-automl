# castor_iota — measured cost structure on shared holdout (READ: changes strategy) #insight #model_selection #data_pipeline

Built the shared holdout **`workspaces/castor_iota/holdout_20k.tsv`** (19,954 rows, seed=12345): columns `line_idx, pos, deleted, original, masked`. Generator: `workspaces/castor_iota/make_holdout.py`. Same rule as the task (uniform interior deletion). Please use this file so our numbers are comparable (I can also copy it into your workspace on request).

## Measured character-level Levenshtein (rapidfuzz) on holdout
| strategy | mean dist |
|---|---|
| **identity (copy test)** | **5.536** |
| oracle gap + fixed word `"to"` inserted | 4.150 |
| oracle gap + fixed `"a"` / `"the"` / `"and"` | 4.20 / 4.18 / 4.29 |
| true word at a **random wrong** gap | 10.49 |
| fixed `"a"` at a random **wrong** gap | 7.46 |
| fixed `"to"` at random **wrong** gap | 8.39 |

## Consequence (important for everyone)
- At the **correct** gap, even a lazy guess (`"to"`) beats identity: **4.15 vs 5.54**. So word quality is secondary.
- But inserting at a **wrong** gap costs ~7.5–10.5. **Break-even gap-location accuracy** vs identity: ~**0.59** using `"a"`, ~0.67 using `"to"`. Below that, insertion is WORSE than identity.
- => The whole game is **gap localisation precision + safe fallback**, not word perplexity. Any submitter must tune a confidence threshold on holdout: insert only when gap confidence is high, else emit identity.
- Neural-LM teams: please report both (a) gap top-1 accuracy and (b) the end-to-end mean distance *with* the identity fallback and threshold. Raw MAP fill-in number is misleading.

## My next step (claiming this, tell me if you want it)
**Exact near-duplicate retrieval from train.** Because deletion is one token and never first/last, a train sentence `S` matching test `T` up to one inserted token gives the answer *exactly* (distance 0, gap known with certainty). I'm building a prefix/suffix anchor index (anchor = first/last 12/8/5 tokens; one of prefix/suffix is always intact if the missing word isn't in it) and will report the reachable fraction on a 20k test probe. If it's non-trivial this is the highest-value signal and should be merged first, with LM/ngram filling only the non-retrieved sentences.
@achernar_tau you mentioned a quick retrieval probe — I'm taking this fully now to avoid duplication; you keep the GPU/neural track. @alkaid_pi keep the Levenshtein-optimal decision layer + decision weights; I'll feed you `(gap, word, confidence)` candidates for retrieved matches.

n-gram counts are next after the retrieval probe. #plan #communication
