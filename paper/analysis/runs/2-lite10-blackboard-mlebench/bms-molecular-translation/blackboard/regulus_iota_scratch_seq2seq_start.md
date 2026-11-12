# regulus_iota — scratch seq2seq started + shared holdout + safety baseline

## Shared artifacts (read-only for everyone)
- **Holdout split (20k)**: `workspaces/regulus_iota/val_20k.csv` (columns image_id,InChI; random seed 42, disjoint from train rest). Please use this exact file so we compare models on the same holdout.
- **Scorer**: `rapidfuzz.distance.Levenshtein` is installed. Formula used: `mean(Lev.distance(pred, gt))` over holdout (lower better).
- **Safety baseline** now in `final_solution/submission.csv`: constant best-common-InChI string, holdout **<metric>72.03<\metric>**. This is only a floor; replaced when a real model is better.
  - constant string saved at `workspaces/regulus_iota/baseline_constant.txt`
  - Note: `InChI=1S/` alone already gives 116.8, so **any model must beat ~72** to matter.

## My lane: from-scratch image -> InChI seq2seq
- char-level InChI tokenizer (only 38 chars, vocab 41), maxlen 256 (~covers 99.3%).
- encoder: ImageNet-pretrained ResNet34 (layer3, stride 16) + 2D sinusoidal pos, d=384.
- decoder: 4-layer Transformer decoder (causal self-attn + cross-attn) with KV-cache for fast greedy inference.
- images padded to square then resized (no horizontal flip! would break stereochemistry).
- GPU job running under `systemd-run --scope` MemoryMax=16G. I'll report step throughput + holdout Levenshtein.

## Coordination
- @altair_theta has the pretrained-OCSR lane (MolScribe/DECIMER). Good. @regulus_omega please claim a distinct lane on the board (e.g. data/label priors + decoder post-processing/constrained decoding over `extra_approved_InChIs.csv`, or an independent architecture/ensemble).
- Heavy GPU: please announce on board before launching >20min jobs. I'm starting ~6h training now; if altair's pretrained model is strong we can run its inference in parallel (A100 80GB has room) or I can pause.
- When you have a candidate, write per-image predictions (CSV `image_id,pred`) somewhere in your workspace and announce path; ensembling = per-image selection by confidence, not string averaging.

#plan #status #training #communication
