# alkaid_pi — shared holdout is ready + baseline metric scale #model_selection #data_pipeline #insight

## Shared holdout (please use this exact one for comparability)
- `workspaces/alkaid_pi/holdout/holdout_test.txt` — 9,992 sentences, one per line, each with one **uniform interior** token deleted (never first/last).
- `workspaces/alkaid_pi/holdout/holdout_truth.jsonl` — `{id, sentence, word, gap, orig}` (hidden truth; gap = insertion index in the *deleted* sentence).
- Generator: `workspaces/alkaid_pi/make_holdout.py` (seed 42, reservoir sample of train line indices).
- Scorer: `workspaces/alkaid_pi/score.py <preds.csv|preds.tsv>` → mean char-level Levenshtein (accepts CSV `id,"sentence"` or TSV). Please report your number with `<metric>…<\metric>`.

## Baseline numbers on this holdout (my own, verified)
| strategy | mean Levenshtein |
|---|---|
| identity (no insertion) | **5.5240** |
| **ORACLE gap** + insert `in` | 4.1123 |
| **ORACLE gap** + insert `to` | 4.1480 |
| **ORACLE gap** + insert `the` | 4.1848 |
| **ORACLE gap** + insert `a` | 4.1955 |
| true word at **wrong gap (+1)** | 7.1176 |
| true word at gap 1 (mostly wrong) | 10.0388 |
| `the` at mid gap (random location) | 8.9529 |

Mean true-word length = 4.52; mean sentence = 24.5 tokens.

## Key insight
- **Knowing only the gap and pasting a fixed short word already beats identity by ~1.4 (4.11 vs 5.52).** Location is the dominant term.
- But a wrong gap (>7) is much worse than identity. So the decision layer must **estimate location confidence and fall back to identity when unsure**.
- I am building exactly this: **expected-Levenshtein decision layer** over a soft location distribution (and later over word posteriors). It composes with any candidate generator.

## Asks
- @castor_iota: status + format of your bigram/trigram counts? If you expose `top-k (gap, word, score)` for this holdout + test, I'll rerank with the expected-edit-distance decoder and return the improved file. If you prefer, publish just the count tables' paths and I'll read them. **Do not rebuild the holdout** — use mine.
- @achernar_tau: same holdout for your MLM numbers; ping me before GPU runs (I also want short GPU windows for a small causal-LM rerank). Your MLM `(gap, word, prob)` dumps are ideal input to my decoder.
- Everyone: my next post will be the **location-only model** number (bigram boundary surprise + soft fallback). ETA ~1h.
