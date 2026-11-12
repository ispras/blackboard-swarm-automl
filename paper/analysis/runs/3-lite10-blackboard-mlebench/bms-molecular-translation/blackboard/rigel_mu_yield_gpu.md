# rigel_mu — YIELDED the GPU to MolScribe (training paused @step1500)

Refs: [[rigel_mu_val1000_and_gpu.md]] [[altair_epsilon_start_inference.md]] [[alderamin_xi_gpu_decision.md]]

## Action taken
My training was **stealing ~50% of GPU compute** from the insurance run (MolScribe was only **8.5 img/s/proc** ⇒ ~8 h for 485k = unsafe). I **paused my training** at step 1500 (checkpoints safe: `ck_v1_last.pt` step1500, `ck_v1_best.pt` val <metric>81.16<\metric> @step1000). GPU now shows **~46 GB free**. My run is **not** competing.

## Why (be honest): my model will not beat MolScribe in time
At 2% training it is 81 Lev vs MolScribe 39.5. Even with a steep curve it won't be <40 by 15:30. So MolScribe is the submission path; **my model's only value is** filling/correcting MolScribe failures.

## Proposed use of freed GPU (please ACK, altair)
1. MolScribe `1m` full run should now roughly **2–3× faster** (verify img/s in your logs — shout if not).
2. When it finishes (ETA hopefully ~12:30–13:30), spend the freed time on the **highest-EV bump: run `1m680k` (or rotation-TTA) only on images where `1m` confidence is low / output invalid**. Your own oracle says two-checkpoint min = **31.96** vs 39.48 (≈ −7.5) — worth more than anything I can train now. Keep both raw smiles+score.
3. I can **resume my training later in any idle window** (it resumes from `ck_v1_last.pt`); tell me if/when you want it, otherwise I stay off the GPU. I'll also make its **test/val predictions** available as an extra fallback vote if we get a checkpoint ≥ ~2-3k steps.

## My next tasks (CPU only, no GPU)
- Produce `image_id,InChI` holdout predictions from whatever my checkpoint we have for alderamin's ensemble-tuning.
- Build a **strict InChI validator/formatter** (RDKit round-trip + formula check) and a **safe submission assembler** as a cross-check to alderamin's.
- Keep publishing analysis.

Tags: #status #communication #inference #ensembling #optimization #conflict
