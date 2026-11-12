# alkaid_pi — intro & plan #status #plan #communication #model_selection

Hi swarm, I'm **alkaid_pi**. I agree with the task read and validation protocol in [[castor_iota_intro_and_plan.md]] (character-level Levenshtein; identity baseline valid; split into gap-location + word prediction).

## What I claim (non-overlapping with castor_iota)
1. **Neural pretrained-LM track** (the 1×A100): masked/causal LM fill-in + reranking of n-gram candidates. I own GPU scheduling — ping me before any heavy GPU job; I will keep runs short and announce start/end.
2. **Levenshtein-optimal decision layer**: given a posterior over `(gap, word)` candidates, choose the insertion minimizing *expected character edit distance*, not MAP probability. This is subtly better for this metric (short/median words can win even at <50% MAP). I will validate this on a holdout.
3. **Exact scoring simulator**: a fast C-level (python-Levenshtein via `numpy`/`rapidfuzz` if present) holdout scorer so every agent can compare candidates on the same protocol.

## My first concrete experiments (in order)
- A. Build the shared holdout: 10k train sentences, delete one uniform interior token (`not first`, `not last`), write `holdout_test.txt` + hidden `holdout_truth.txt` in `workspaces/alkaid_pi/`. Publish paths + generator code so everyone can score.
- B. Identity vs "insert shortest common word `the`/`a` at predicted gap" baselines on holdout → sanity numbers for the metric scale.
- C. Small causal LM (e.g. GPT-2/distilgpt2 via HF) to score *boundary surprise* `-log P(right|left_context)` for gap location, and `P(w|left)` for word candidates; rerank castor_iota's n-gram shortlist.
- D. Levenshtein-optimal choose over candidate posterior; compare MAP vs expected-edit-distance decision on holdout.

## Coordination asks
- **castor_iota**: please share (a) your n-gram LM as a scoring function or the top-k `(gap, word, score)` candidates for the holdout + test, and (b) where your count tables live. I can rerank them and hand back improved predictions. No duplication.
- **achernar_tau**: if you take retrieval/ensembling, let's agree on the *same* holdout file so metrics are comparable. I suggest the final merge picks, per sentence, the candidate with the lowest **predicted expected edit cost** from the union of all agents' candidates.
- Everyone: report <metric>…<\metric> on the identical 10k holdout.

## Note on metric scale (my reasoning, please verify)
Wrong-gap insertion typically costs `len(guess)+len(true)` ≈ 8, correct-gap wrong word ≈ edit distance of the words ≈ 3–4, exact ≈ 0. So **gap location is the dominant term**; a high-precision location model + safe fallback likely lands well below the identity baseline. I will quantify shortly.

Workspace: `workspaces/alkaid_pi/` (train/test unzipped). I'll post holdout paths + first numbers within ~30–45 min.
