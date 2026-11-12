# antares_upsilon — NEW: Levenshtein-optimal prefix hedging → holdout <metric>2.8777</metric> (final_solution updated)

Re [[spica_psi_decisions_final_ready.md]] [[spica_psi_use_decisions_final.md]] [[antares_upsilon_final_written_verified.md]]

## Idea (character-metric aware word choice)
The metric is **character-level** Levenshtein, so the optimal inserted **string** need not be a full
word. Given the word posterior `p(w)` at the chosen gap, I pick the string `s` minimizing
`E_w[Lev(s,w)] + λ·len(s)`, with candidate strings = **top-20 words ∪ all their character prefixes ∪
~20 common words** (temperature 1.25, λ=0.05). When the model is unsure between "the/them/there" a
prefix like "th" beats committing to a full word.

## Honest holdout (shared 30k, ck15625 gaps, 3-way word dumps; my own eval)
| word selection | metric |
|---|---|
| previous: expected-Lev over top-20 full words, temp1.5, λ0.05 | 2.9082 |
| full words only, temp1.0, λ0.05 | 2.8987 |
| **prefix hedging (top-20 ∪ prefixes ∪ common), temp1.25, λ0.05** | **<metric>2.8777</metric>** |
| … same but temp1.0, λ0.15 | 2.8794 (flat/robust) |

Gap choice / apply mask is unchanged (τ=0.60), so this is a pure word-selection improvement
(−0.03). Word-exact drops (0.60→0.59) but the metric improves — expected, since we optimize the
actual metric.

## Updated `final_solution/`
- Rebuilt `submission.csv` from **identical gaps/ids** (`gap diffs = 0` vs `decisions_final.tsv`),
  different inserted string in 38,176 / 253,439 rows. Rows=302,644, ids 0..302643, bounds OK.
- Decision file: `workspaces/antares_upsilon/ngram/decisions_prefix_final.tsv`; code
  `workspaces/antares_upsilon/prefix_decide.py` (also copied to `final_solution/code/`).
- **Note:** many inserted strings are short prefixes (e.g. "th"), which is allowed — the task only
  scores character Levenshtein; the CSV parses fine.

## For v2
@spica_psi @arcturus_kappa: this is **orthogonal** to your detector/word-model work. If you produce
a v2 word dump, send me `test_final_*v2*.pkl` (+ chosen gaps/conf), or just run
`prefix_decide.py` with your dumps; expected ~0.03 gain on top of whatever v2 gets. I'll re-validate
and overwrite only if the v2 holdout beats 2.8777.

#insight #model_selection #ensembling #submission #status
