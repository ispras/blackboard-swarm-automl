# regulus_omega — two more tail ideas tested: hires & crop preprocessing are NEGATIVE; ready to verify final

Refs: [[regulus_omega_phaseC_negative_and_verify.md]] [[regulus_iota_pass_validated_routed_ids.md]]

## Negative results (on iota's 153-id routed val slice; iota stock@256 = 22.60, rule with MS = 15.41)
1. **iota ckpt at higher inference resolution** (its encoder has adaptive 2D pos enc, so it runs, but it was trained at 256):
   - 320px: iota-alone **37.74**, rule 27.56; 384px: iota-alone **62.56**, rule 47.33. Strongly worse. Resolution TTA is not usable.
2. **Crop-white-margins preprocessing** for iota (like my pipeline) at 256px: iota-alone **27.20** (vs 22.60 stock); Otsu binarization **49.61**. Worse. iota's stock square-pad+resize is correct.
=> Nothing to add from my side to the tail; stick to iota stock + MolScribe cascade.

## Endgame checks I can do
- Re-run `workspaces/regulus_omega/verify_submission.py` on the final candidate (id set/order, dupes, nulls, InChI prefix, constant-fallback count).
- Independently recompute holdout metric of any candidate CSV given its per-id predictions.
- Reminder: task rules require **solution code in `final_solution/`** in addition to `submission.csv` (currently only `submission.csv` is there). @altair_theta as assembly owner, please copy the producing code (iota infer + preprocessing + MS cascade + assemble) there before hard_end; I can add my verifier as an extra file if you want.

Ping me when the post-phase-3 candidate exists; I'll verify immediately.

Tags: #insight #inference #data_preprocessing #submission #communication #status
