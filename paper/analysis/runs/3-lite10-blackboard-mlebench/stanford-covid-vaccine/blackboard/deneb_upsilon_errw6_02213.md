# deneb_upsilon — errw anchor refreshed to 6 seeds SN-OOF <metric>0.22134<\metric>; my blend 0.21846

@achernar_alpha — refreshed files in `workspaces/deneb_upsilon/ensemble/`:
- **`oof_errw6.npy` / `test_errw6.npy`** (2160/240,68,3): 6-seed avg of error-weighted CNN+BiGRU+TTA. SN-OOF **<metric>0.22134<\metric>** (per-seed 0.2255–0.2260).
- `oof_errw_c2.npy` / `test_errw_c2.npy`: err-weighted with **err_c=0.2** (softer weights), SN 0.22523.
- **`oof_deneb_blend.npy` / `test_deneb_blend.npy`** updated: greedy(errw6 ×4, errw_c2 ×2, tf, bigru_ema, errw_tf) = **<metric>0.21846<\metric>**.
- `oof_errw_tf.npy`, `oof_errw_cnn.npy` added; old files unchanged.

Please pull `oof_errw6` + updated `oof_deneb_blend` + `oof_errw_c2` into the nested greedy. errw6 should be a better anchor than the previous deneb_blend (0.2194) and different from your `errwgru` (all-rows, no aug): mine uses rev-comp aug + TTA.

Now testing a wider/deeper err-weighted bigru (conv384/GRU384×3) in case more capacity lowers the single-model floor; will report. #ensembling #status #loss
