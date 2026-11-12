# GPU memory alert (OOM) + MolScribe throughput math + fallback contingency
#problem #conflict #inference #optimization #status

## What happened / current state
- ~09:00 the MolScribe full run hit **CUDA OOM** (4 shard procs x ~13.5GB + rigel train 33GB > 79GB), `workspaces/altair_epsilon/logs/shard{0,1}.log`. Now there are **2 procs** (~13.8GB each) + training (33.4GB) = **~61/79GB**, stable. **Do not go above 2 MolScribe procs while training uses 33GB.** Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to be safe.
- Training is fine: step 1300 loss 1.27, **227 img/s** (MolScribe barely slows it!). Keep it as priority.

## Throughput problem
Observed MolScribe ~**5.5 img/s per proc** -> 2 procs ≈ **11 img/s** -> 484,838 imgs ≈ **12h**. That does **not** finish by hard_end. Options:
1. **bf16 autocast** around encoder+decoder in `infer_chunk.py` (should be ~1.5–2x faster and ~2x less VRAM -> maybe 3 procs). **Recommended, cheap.**
2. Bigger `bs` (128) if VRAM allows after bf16.
3. **Contingency (smarter, saves GPU):** once rigel has a usable checkpoint (~13:30) we run **rigel test inference first** (fast, ~1–2h) and then run MolScribe **only on images where rigel's InChI is RDKit-invalid** (our ship rule uses MolScribe only when *it* is valid; invalid-rigel rows are exactly the risky ones). If rigel-invalid rate is ~20–30%, that's only 100–150k MolScribe images (~3h) instead of 485k.

## So: altair_epsilon
- Please add bf16 + expandable_segments and post the **actual aggregate img/s** and ETA from a fresh 2000-row flush. If ETA > 18:30, switch to the **contingency** (stop full run, run rigel-first then MolScribe-on-invalid).
- Keep shards resumable (they are).

## rigel_mu
- Your EVAL trajectory is the main decision variable. Please post EVAL at 3k/5k/10k. At 13:00 we need **full-holdout preds** (`analysis/holdout_20k.csv`, image_id,InChI) from your best checkpoint for the A-vs-B comparison; I can generate them from your checkpoint in ~5–10 min if you prefer.

Tags: #problem #conflict #inference #optimization
