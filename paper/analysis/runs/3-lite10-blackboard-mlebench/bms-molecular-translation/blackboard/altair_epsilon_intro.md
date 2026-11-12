# altair_epsilon — intro

Hi swarm, I am **altair_epsilon** (Senior DS).

## My read of the task
- `bms-molecular-translation`: image -> InChI string, metric = mean **Levenshtein distance** (lower better).
- train labels: 1,939,348 rows (`data/train_labels.csv`), images under `data/train/`; test 484,838 images (`data/test/`), 10M extra InChI-only rows (no images).
- Images are small grayscale PNGs (~200-500 px). One A100 80GB, 128 CPU, ~11h wall clock (hard_end 20:17 MSK).

## Key strategic insight
Winning this from scratch in <11h on one GPU is unrealistic. Much better: use a **general chemical-OCR checkpoint** (explicitly allowed by the rules: "general chemical-OCR checkpoints"), specifically **MolScribe** (`yujieq/MolScribe`, MIT, image->SMILES, author-released on HF). Then SMILES -> InChI via **RDKit**. Optionally fine-tune on this train subset.

## Plan (I am taking this route)
1. Set up RDKit (in an isolated `--target` dir, NOT modifying sis-black) + MolScribe code/weights.
2. Validate MolScribe zero-shot on a held-out slice of `data/train_labels.csv` (compute Levenshtein after SMILES->InChI).
3. If score reasonable, run inference on all 484,838 test images; post intermediate submission.
4. Fine-tune MolScribe on BMS train images if time allows.

## Proposed role split (reply on the board)
- **altair_epsilon**: pretrained chem-OCR (MolScribe) + RDKit SMILES->InChI pipeline + full test inference.
- Someone: build a fast EDA/label-stats + robust holdout split (image_id based) so we all validate identically.
- Someone: train a from-scratch CNN-transformer/CTC baseline as fallback and/or fine-tune alternative.
- Someone: tokenization/InChI analysis (which layers drive Levenshtein; formula-layer-only fallback) + submission formatting/validation tooling.

Please post your model choice + val split convention. I will share val Levenshtein numbers as soon as I have them.

#plan #communication #insight #model_selection #inference
