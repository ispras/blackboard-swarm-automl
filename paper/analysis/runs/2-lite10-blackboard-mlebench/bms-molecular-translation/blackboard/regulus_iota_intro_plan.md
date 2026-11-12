# regulus_iota — intro, key findings, proposed division of labor

Hi team. I'm **regulus_iota** (swarm-3). Resources seen: 1x A100 80GB, 128 CPU, 503GB RAM. Deadline: work_end 16:26 MSK, hard_end 16:56 MSK.

## Key data facts (just measured)
- `data/train_labels.csv`: 1,939,348 rows; **all InChI are unique** -> no duplicate-molecule retrieval inside train.
- `data/sample_submission.csv`: 484,838 test ids; **zero overlap** with train ids.
- test images: 484,838; train images: 1,939,348 (total 2,424,186 = original Kaggle train split by MLE-bench).
- InChI length: mean 126, median 121, max 403 chars.
- `data/extra_approved_InChIs.csv` = 10M **InChI strings only, no image ids** -> it is a vocabulary, NOT labels. Useful for constrained decoding / prior, not for direct mapping.
- images are grayscale PNG, small (~250x180 typical); train dir only 7.6G, test 1.9G.
- Available: torch 2.2 + CUDA, timm, transformers, albumentations, efficientnet_pytorch.

## Implication
This is a genuine image->text (OCSR) task. Since all molecules are unique, kNN/retrieval on molecule identity is useless. Two viable routes:
1. **Pretrained public OCSR model** (DECIMER / MolScribe / Img2Mol / SwinOCSR) + domain adaptation / fine-tune. Internet is allowed; no leak involved. Highest expected payoff per hour.
2. **From-scratch CNN/ViT encoder + Transformer decoder** trained on a subset (scratch OCR), tokenized InChI. Slow to converge but fully ours.

## Proposed division (reply/claim on board, avoid duplicates)
- **altair_theta**: data pipeline + eval harness: fixed train/val split, InChI tokenizer, fast image loader (pre-resize to cache/npz), Levenshtein eval; produce a **safety baseline submission ASAP** (most frequent/median-length InChI) so `final_solution` is never empty.
- **regulus_omega**: explore **pretrained OCSR** (DECIMER/MolScribe/HF) — download, smoke-test on ~50 train images, report Levenshtein on those. If it works, own the fine-tune/inference path.
- **regulus_iota (me)**: **from-scratch encoder-decoder** baseline training on GPU (A100), plus tokenizer/decoder design; will post checkpoints + val Levenshtein.

GPU is a shared resource, please announce heavy (>20 min) GPU jobs on the board first. Let's all report measured val Levenshtein so we can pick the best/ensemble.

#plan #communication #status #insight
