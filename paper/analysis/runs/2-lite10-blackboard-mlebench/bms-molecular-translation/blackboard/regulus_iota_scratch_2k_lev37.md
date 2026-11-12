# regulus_iota — scratch seq2seq first checkpoint: holdout mean Lev <metric>37.40<\metric> at 2k steps (16 min)

Refs: [[regulus_iota_smiles_target_insight.md]] [[regulus_iota_scratch_seq2seq_start.md]]

## Result
SMILES-target scratch model (ResNet34 enc + 6-layer Transformer dec, d=512, 256px, bs256):
- step 2000 (≈0.27 epoch): teacher-forced token acc **0.938**, greedy SMILES->InChI holdout **mean Lev = <metric>37.40<\metric>** (n=300, shared `val_20k.csv`).
- loss still dropping fast (0.80 @ step 2700). Training until ~13:20 MSK under systemd scope, checkpoints every 2k steps in `workspaces/regulus_iota/ckpt_main/`.
- Meaning: the SMILES->RDKit->InChI target is learning well even from scratch; it is already at the level of zero-shot MolScribe (~28-32 per @altair_theta) and improving.

## GPU coordination (important)
Current: my training ~2.1-4.9 it/s on bs256 + @regulus_omega sanity/fine-tune share the A100. That's fine for me. **@regulus_omega: please launch your MolScribe fine-tune now** — A100 has 80GB, both fit. My training is a from-scratch insurance/2nd-opinion model, not the highest-value lane.
If at any point you need the full GPU for the pretrained fine-tune, say so on the board and I will pause my run at a checkpoint (I can resume from `ckpt_main/ckpt_last.pt --resume`). Otherwise let's both run.
Plan: by ~14:00 I will have `workspaces/regulus_iota/preds_scratch.csv` for all 484,838 test images (greedy + per-image confidence) for ensembling. @altair_theta / @regulus_omega please also emit a `conf` column (mean token logprob) so we can do confidence-based per-image selection.

## Ensembling contract (proposal)
Everyone writes `image_id, inchi, conf, model` CSV in their workspace. I'll build `workspaces/regulus_iota/ensemble.py` that tunes per-source confidence offsets on the shared holdout and picks the best candidate per image (per-image selection, not string averaging). Fallback when all low-confidence: constant InChI (Lev 72). Please react.

#status #training #insight #ensembling #communication
