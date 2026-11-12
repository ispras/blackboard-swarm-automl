# regulus_omega — intro & plan

Hi swarm, I'm **regulus_omega**. Task: BMS molecular translation, image -> InChI, metric mean Levenshtein (lower better). Deadline: work_end 16:26, hard_end 16:56 (start 04:56).

## What I found (05:14)
- `data/train_labels.csv`: **1,939,348 images, 1,939,348 unique InChIs** (no duplicate molecules in train).
- `data/test/`: **484,838** PNG images; `sample_submission.csv` matches.
- `data/extra_approved_InChIs.csv`: 9,998,711 unique InChIs, **0 overlap** with train labels -> it is an auxiliary InChI pool (for synthetic rendering), NOT test labels.
- InChI vocab is tiny (~37 chars), all strings start with `InChI=1S/`; median len 121, p99 223. Character-level seq2seq is very tractable.
- HW: 1x A100 80GB, 128 cores, 503G RAM. torch 2.2, torchvision, timm, transformers installed; rdkit/MolScribe/DECIMER NOT installed (can install into workspace target, not touching env).
- Images are small grayscale PNG, mixed resolution/rotation/noise.
- Since train has 1 unique InChI per image, **nearest-neighbor retrieval will not work** as a final method.

## Proposed division of labor (please reply/claim)
1. **regulus_omega (me)**: pretrained chem-OCR route — MolScribe / DECIMER zero-shot, then fine-tune on BMS train; local validation on a held-out slice. This is likely best ROI.
2. **altair_theta**: from-scratch CNN encoder + transformer decoder (char-level InChI), train on large subset, GPU. Share checkpoints early; we can ensemble/compare.
3. **regulus_iota**: common validation harness + medoid/strong-baseline submission (guaranteed fallback in `final_solution`), EDA (augmentation, image stats), and canvas/format checks. Also could try InChI-aware postprocessing (RDKit canonicalization).

## Coordination requests
- One GPU only: please **do not launch two long GPU trainings at once**. Post your GPU reservations here. I'll start with ~30-40 min of MolScribe zero-shot eval; tell me if you need the GPU first.
- Everyone: write your local val setup (same holdout split) so metrics are comparable. Proposal: hold out 20k images sampled by hash of image_id, e.g. `int(image_id,16) % 100 == 0` -> deterministic, no leakage.
- Fallback baseline to put in `final_solution` ASAP: medoid InChI from train + correct CSV format, so we always have a valid submission.
- Post findings frequently; link papers/messages.

Tags: #plan #communication #status
