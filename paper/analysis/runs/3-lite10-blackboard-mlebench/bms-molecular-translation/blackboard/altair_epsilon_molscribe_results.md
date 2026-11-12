# altair_epsilon: MolScribe zero-shot WORKS — 48.5 mean Lev (better than converted path) + inference speed

Refs: [[alderamin_xi_molscribe_postproc.md]] [[alderamin_xi_no_dict_snapping.md]] [[alderamin_xi_shared_infra.md]] [[alderamin_xi_gpu_schedule.md]]

## Result (first 256 train rows; will re-run on `analysis/holdout_20k.csv` now)
MolScribe `swin_base_char_aux_1m.pth` zero-shot, greedy, 384px, **without** fine-tuning.
Key trick: **do NOT use `predict_images()`'s `convert_graph_to_smiles`** (it returns `<invalid>` / `*` for 44%!). Instead decode directly and take the raw `pred['chartok_coords']['smiles']`, then postprocess.

| variant | mean Lev | exact | empty |
|---|---|---|---|
| converted (altair's first run) | 65.29 | 33.6% | 44.1% |
| raw chartok smiles | 61.40 | 28.5% | 40.6% |
| raw + keep largest fragment | 49.18 | 37.9% | 27.3% |
| **raw + largest + replace `*`→C** | **48.52** | 37.9% | 26.6% |
| oracle min(raw_frag, converted) | 47.82 | 43.0% | - |

=> raw decoding + fragment cleanup beats the converted path by ~7 Lev. Please use the raw path in any postproc code.

## Inference speed (shared GPU, rigel training also running)
- 1 proc bs=64: **11.7 img/s**; 3 procs bs=64: **~25 img/s aggregate** (GPU sharing with rigel).
- Full test = 484,838 imgs -> **~5.5 h with 3 procs**. Must start by ~13:30 to be safe.

## Proposal (please ACK/object now)
I start **full 485k MolScribe zero-shot inference NOW, sharded 3 procs**, writing `image_id,smiles` shards -> then RDKit postproc -> submission. This is our safe strong submission (~48 expected, far better than any constant/scratch baseline in the time left).
- **alderamin_xi**: please do NOT launch a duplicate MolScribe full run; instead take the **postproc + `analysis/eval_lev.py` validation of my shard outputs** and the submission assembler/validator. I'll give you the shard file paths.
- **rigel_mu**: keep training but expect ~2x slower epochs while 3 inference procs run. If your val at ~12:30 is not clearly <40, we ship MolScribe. Please post val number by then.
- Also: I downloaded `swin_base_char_aux_1m680k.pth`; will compare on the holdout and use the better one.

Tags: #inference #postprocessing #insight #status #plan #model_selection
