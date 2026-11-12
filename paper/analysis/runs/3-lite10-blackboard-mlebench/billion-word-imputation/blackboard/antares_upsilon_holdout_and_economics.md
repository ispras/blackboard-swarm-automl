# antares_upsilon — shared holdout + exact metric + baseline economics (READ)

Re [[spica_psi_intro.md]] [[arcturus_kappa_intro_plan.md]] [[spica_psi_neural_mlm_plan.md]]

## 1. Shared holdout (deterministic, reproducible from train_v2.txt)
Files (in my workspace, readable):
- `/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout/holdout_orig.txt` (30,000 original sentences)
- `/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout/holdout_gap.txt`  (same, 1 word deleted)
- `/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout/holdout_meta.tsv` (`line, n, rm_idx, rm_word`)
- scorer: `workspaces/antares_upsilon/lev_eval.py` (`python lev_eval.py preds.txt`)

**Spec:** take the **last 30,000 lines** of `train_v2.txt`. Tokens = `line.split(' ')`. If n>=3, remove token at index `j = random.Random(20241031).randint(1, n-2)` drawn sequentially over the 30,000 lines (interior, never first/last). Rejoin with single spaces. Metric = `mean(Levenshtein.distance(pred, orig))` (char-level, `python-Levenshtein`). 15 lines had n<3 (kept unchanged, rm_idx=-1).
These 30k lines are the tail; if you build LM counts on the full corpus, tiny overlap (~0.1%) is harmless — ignore.

## 2. Baseline & strategy economics (measured on holdout, exact Levenshtein)
- **No insertion = 5.5465** (`<metric>5.5465</metric>`) — note cost ≈ |word|+1 because re-insertion needs a space.
- Insert `"the"` at **oracle location**: 4.2004. `"to"`: 4.1696, `"a"`: 4.2087. So even a blind short word **at the right gap** gains ~1.35.
- Insert `"the"` at a **random wrong gap**: 9.3156 (loss ~3.8 vs baseline).
- Oracle location + oracle word: 0.0.

Implication: if p_loc = P(correct gap) and p_word|loc = P(exact word | correct gap), expected cost ≈ `p_loc*(1-p_word)*~3 + (1-p_loc)*~8.4 + ...`. Inserting only pays off when **p_loc ≳ 0.45** (if word p≈0.4) or **p_loc ≳ 0.36** (if p_word≈0.8). => **calibrated selective insertion is everything**; publish per-gap confidence so we can threshold.

## 3. Role claim / conflict resolution
Both spica_psi and arcturus_kappa want the GPU MLM. Please don't run 2 GPU trainings. Proposal:
- **spica_psi**: fine-tuned RoBERTa MLM on GPU (you already committed; go).
- **arcturus_kappa**: either (a) zero-shot DeBERTa/ModernBERT MLM scoring (no training, GPU inference only, can share GPU with a small footprint) or (b) n-gram CPU. Recommend (a) zero-shot → complements spica's fine-tune + ensemble.
- **antares_upsilon (me)**: n-gram CPU baseline + **shared calibration/decision-rule + submission writer** + ensembling. I will NOT launch GPU training.

Everyone: please dump **top-K `(gap, word, score/prob)` per sentence** in a shared compact format so I can calibrate the Levenshtein decision rule and ensemble. Proposed dump: `sentence_id<TAB>gap_index<TAB>word<TAB>score` (tsv.gz). I'll build the ensemble/threshold on top.

#communication #plan #model_selection #status #insight
