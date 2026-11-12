# BMS Molecular Translation — swarm solution

Final submission: `submission.csv` (`image_id,InChI`), produced from the swarm's best model.

## Pipeline (winner: regulus_iota scratch image->SMILES seq2seq)
1. `iota_common.py` — model definition (ResNet34 encoder at 256px letterboxed/padded image,
   6-layer Transformer decoder d=512, char-level SMILES vocabulary, KV-cache greedy generate).
2. `infer_iota.py` — batched, resumable full-test inference over `test_ids_sorted.csv`
   (images sorted by PNG size for batch-length homogeneity). Emits
   `image_id,smiles,conf,inchi_from_smiles,valid`, converting SMILES->InChI with RDKit.
3. `tta180.py` — optional 180-degree rotation TTA on the low-confidence subset.
4. `combine.py` — confidence-gated per-image selection between base and TTA/second-opinion
   candidates (threshold tuned on the shared 20k holdout), constant-InChI fallback for
   missing/low-quality rows. Writes the final `image_id,InChI` CSV.
5. `predict.py` — (swarm) MolScribe zero-shot second opinion: aspect-preserving letterbox,
   chartok graph decoding -> SMILES -> RDKit InChI.

## Reproduce
```
# winner checkpoint test pass (regulus_iota ckpt)
python infer_iota.py --ckpt <ckpt_best.pt> --out preds_iota_test.csv --batch 256 --workers 8
# optional second opinion on low-conf ids
python tta180.py --ckpt <ckpt_best.pt> --ids_csv lowconf_ids.csv --split test --out preds_tta180.csv
# final assembly
python combine.py --main preds_iota_test.csv --tta preds_tta180.csv --sweep --out submission.csv
```

Validation: shared holdout `workspaces/regulus_iota/val_20k.csv` (20,000 images held out from
`data/train_labels.csv`); metric = mean Levenshtein distance between predicted and true InChI.
