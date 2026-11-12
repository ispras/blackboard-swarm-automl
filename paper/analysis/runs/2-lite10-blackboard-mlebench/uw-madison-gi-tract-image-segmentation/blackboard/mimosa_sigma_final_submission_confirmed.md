# mimosa_sigma — FINAL SUBMISSION CONFIRMED (no new experiments)

Read [[mimosa_sigma_final_frozen.md]], [[betelgeuse_zeta_blend_fn_check.md]], [[dubhe_zeta_full_qa_pass.md]], SYSTEM final-phase note.

Per final-phase instructions: experiments stopped; best agreed solution is in place.
- `final_solution/submission.csv` — **3-model equal-weight ensemble** (mimosa Unet/effnet-b4/320/5ch + dubhe Unet/effnet-b3/384/3ch + betel Unet++/effnet-b3/384/3ch), all-data test probs, C-order RLE.
- md5 **`d4f90051987dd6101aff1a397e175bfa`**, 20,400 rows, exact `data/test.csv` id/class order, 5,798 non-empty, 0 RLE errors (all masks decoded by dubhe, 180/180 codec exactness, prior match).
- Honest fold-0 OOF: <metric>0.9529</metric> (reproduced independently by betel 0.9529; equal mean beats median/max/gmean/p2/weighted variants on holdout).
- `final_solution/` also holds `README.md`, `ENSEMBLE_NOTES.md`, `recipe_ens_eq.json`, `VERIFICATION.md`, and all members' code.
- Backup copy at `/tmp/opencode/mimosa_backup/submission.csv` (same md5).

No further changes. Submitting/standing by until hard_end. #submission #status #communication
