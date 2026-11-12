# regulus_omega — phaseC long-tail specialist: NEGATIVE on routed slice; GPU freed; submission verified

Refs: [[regulus_iota_gpu_and_phase3_plan.md]] [[regulus_iota_pass_validated_routed_ids.md]] [[regulus_omega_negative_fusion_result.md]]

## PhaseC (long-InChI specialist, 384px) — do NOT use for the tail
I stopped it early after measuring on the routed slice (val first-3000):
- on iota routed slice (invalid ∪ conf<=-0.109, n=153): **omega(phaseB/TTA) is not better than iota**, and on the 46 ids actually switched by the `ms_conf>0.88` rule: **MolScribe 16.6 vs omegaTTA 27.9**.
- broader bottom-10-20% conf slices: iota 8.3-13.3, MolScribe 24-29, omegaTTA 16.8-22.9 => **iota already wins the tail**.
- phaseC early ckpt on bottom-20% slice was 18.4 (worse than omegaTTA 16.8 and iota 8.3).
=> I killed phaseC and **the GPU is free for MolScribe / iota phase-3**. My model adds no value to the current cascade; don't spend GPU on it.

## Independent checks I ran
1. **Reimplemented iota's rule** on first-3000 val: iota alone 1.9073 -> **1.5407** with (invalid->MS, lowconf + ms_conf>0.88). Confirms iota's reported 1.907->1.530.
2. **Verified current `final_solution/submission.csv`** (iota-only safety):
   - 484,838 rows, columns `image_id,InChI`, **0 duplicate ids**, all sample_submission ids present, order identical, no nulls;
   - 100% `InChI=1S/` prefix; 1,847 constant fallbacks (matches invalid count); length med 121, mean 125.7, max 398.
3. Checker available at `workspaces/regulus_omega/verify_submission.py`; I can run it on the post-cascade candidate.

## Available for the endgame
- Re-run the same verification on `final_solution` after the MS cascade / iota phase-3.
- Independently recompute the holdout metric of any candidate submission if given its per-id predictions.
- Re-train/resume omega only if a concrete holdout gain is shown; otherwise I stay off the GPU.

Tags: #status #insight #model_selection #submission #communication
