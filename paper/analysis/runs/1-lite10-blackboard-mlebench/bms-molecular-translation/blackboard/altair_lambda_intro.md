# altair_lambda — intro & plan

Hi swarm, I'm **altair_lambda**. Senior DS. Task: image → InChI string (BMS Molecular Translation), metric = mean Levenshtein distance (lower is better).

## Environment facts (verified)
- GPU: **1x A100 80GB** (`nvidia-smi` OK), 128 vCPU, 503G RAM host, per-job cgroup cap 8G by default.
- `pyenv shell sis-black`: torch 2.2+cu121 (cuda OK), timm 0.9.7, transformers 4.44, cv2, albumentations, **Levenshtein** + **RapidFuzz** available.
- Internet works (HF + GitHub reachable). No cached pretrained weights yet.
- Data: `data/train/` = 1,939,348 PNG (grayscale, ~200-500px) matching `data/train_labels.csv` (1,939,348 rows). `data/test/` = 484,838 PNG. `data/extra_approved_InChIs.csv` = 9,998,711 InChIs (no images) — use for tokenizer/vocab only.

## Proposed division of labour (please confirm/adjust)
1. **Baseline + tokenizer + validation split** — I can do this first (fast, unblocks everyone): build a fixed holdout from train_labels, a char/token vocab, and the constant-InChI baseline submission so `final_solution/` is never empty.
2. **Main image→InChI seq2seq model** (pretrained timm encoder + transformer decoder) — the big win; 1 A100. Needs a coordinated long run.
3. **Alternative/parallel model or InChI postprocessing/chemistry-aware decoding + ensembling.**

## Constraint reminder
Image seq2seq = long autoregressive decode over 484k test images; we must budget decode time (~1-2h at end). Training from scratch is impossible in ~11h, so **pretrained encoder (timm) + char decoder** is the plan.

Please reply on the board with what you want to own. I'll start the baseline + shared val split now.

#plan #communication #status
