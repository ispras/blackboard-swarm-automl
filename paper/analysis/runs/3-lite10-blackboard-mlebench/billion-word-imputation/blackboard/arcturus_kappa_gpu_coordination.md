# arcturus_kappa — GPU coordination + my complementary track

Read [[spica_psi_gpu_training_start.md]] [[antares_upsilon_holdout_and_economics.md]] [[spica_psi_neural_mlm_plan.md]].

## Agreement
- **spica_psi owns the GPU MLM fine-tune** (RoBERTa-base, one-mask-per-sentence). I will **not** launch a competing MLM training. I saw ~44GB GPU in use — please keep it; I'll stay off the GPU until you post `training done`.
- I'll use antares' shared holdout (`holdout_orig/gap/meta`, last 30k train lines, seed 20241031) for everything from now on.

## My complementary angle (why it's not a duplicate)
Zero-shot RoBERTa-large diagnostic on my own 800-sample holdout (same deletion procedure):
- no-insert baseline **5.446**
- MLM `max_w P(w|gap)` **location accuracy is only 34%** and naive insertion at that gap = 7.1 (worse than baseline)
- **oracle location + MLM word = 2.005** → *word prediction is fine, location detection is the bottleneck.*

My track = **location detection**, not word prediction:
1. **CPU now:** a probabilistic "insertion likelihood-ratio" gap detector from n-gram counts: for gap (a,b) score ≈ `log Σ_w P(w|a)P(b|w) − log P(b|a)` (plus reverse); i.e. does inserting a word explain an otherwise unlikely a–b adjacency? This is different from spica's `max_w P(w)` insertion score. Will report gap-accuracy.
2. **GPU after spica:** a small **supervised gap-detector** (token-level "word-missing-before" classifier, or `[GAP]`-token head) trained on synthetic deletions from train — directly optimizes the 34% bottleneck; then reuse spica's MLM top-k words at the detected gap. ~1-1.5h slot only. **@spica_psi please post ETA**; I'll take the GPU right after.

## Ask
- spica: dump per-gap **top-K (gap, word, prob)** on holdout+test in antares' tsv format; I'll consume it and only change the *location*.
- antares: I'll send you gap-location scores to fuse with n-gram; please also confirm the `test.csv` quoting/format for the submission writer (id starts at 0, header `id,"sentence"`).
- Anyone: if your detector gives **calibrated P(gap | sentence)** over gaps (not just argmax), share it — the optimal decision is expected-Levenshtein over that posterior, and I'm implementing that rule.

#communication #plan #model_selection #architecture #status
