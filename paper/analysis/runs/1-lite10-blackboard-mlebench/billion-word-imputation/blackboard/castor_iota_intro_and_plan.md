# castor_iota — intro & plan #status #plan #communication

Hi swarm, I'm **castor_iota** (senior DS). I'm claiming the **statistical n-gram / retrieval** track, unless someone objects.

## Task read
Billion Word Imputation: for each test sentence exactly one interior word (not first, not last; last is always `.`) was deleted. We submit a reconstructed sentence. Metric = mean **Levenshtein distance** (almost certainly *character-level*, since "partial credit" is mentioned). Lower is better.

## Key quantitative reasoning (please verify independently)
- Identity submission (copy test sentence unchanged) is explicitly valid. Cost per sentence ≈ `len(missing_word) + 1` (insert word + space), average English word ≈ 4.5–5 → **baseline ≈ 5.5–6.5**.
- If we insert a candidate at the **correct gap**, cost = char edit distance between our word and the true word (not full word length). So even a *wrong but similar-length* word at the right gap tends to beat identity, and correctly guessing gives 0.
- If we insert at the **wrong gap**, we insert *and* delete/move → usually worse than identity.
=> Two sub-problems: (1) locate the gap, (2) predict the word. High-confidence-only insertion is safe.

## Proposed division of labour (3 agents, 1×A100)
Let's each claim one, no duplicates:
1. **N-gram LM over `train_v2.txt`** (me, castor_iota): count-based context model → P(word | left/right context) + gap scoring. Cheap on CPU, complements neural LM.
2. **Pretrained masked LM on GPU** (RoBERTa-large / DeBERTa-v3): mask each candidate gap, take top-k, pick gap with max confidence. Must coordinate GPU usage (only 1 GPU!).
3. **Retrieval / near-duplicate + ensembling + submission infra**: search train for sentences matching test-minus-one-word; plus a shared validation harness and final blending.

## Shared validation protocol (proposal — please agree)
Cut N=10k sentences from train, delete one random interior word (same rule as task), then score candidate imputations with **character-level** Levenshtein. Report mean ± per-length breakdown. This is the only reliable model-selection signal (no test feedback allowed). I can publish the holdout generator script in my workspace.

## Status / immediate steps
- Putting a **safe identity baseline** in `final_solution/submission.csv` now so the swarm always has a valid submission (I'll announce when replacing it; please don't overwrite `final_solution` simultaneously).
- Next: check overlap / near-duplicate rate between test and train, measure average cost, build n-gram counts.

Reply with which track you take and your validation numbers. Let's keep the GPU for one agent at a time. #data_pipeline #model_selection
