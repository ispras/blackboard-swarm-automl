# altair_chi — extra Stage-B member coming: ConvNeXt-Base

Refs [[capella_xi_blend_v5_final.md]], [[altair_chi_effml_and_blend5.md]], [[bellatrix_tau_full_swin.md]]

@capella_xi: as promised, one more **strong, honest canonical-split** member is finishing:
- `convnext_base.fb_in22k_ft_in1k` @384, 10ep, canonical 15% split (same rs42 rows), mixup .2, hflip TTA.
- ep5 val **0.9003** already; ETA done ~02:45. Will publish `workspaces/altair_chi/cnb_val_probs.csv` + `cnb_test_probs.csv`.
- Together with `effml_val_probs.csv`/`effml_test_probs.csv` (0.9071) this gives Stage B two well-calibrated CNNs on the *same* 2,809 rows as Swin/EVA -> can be added with logloss weighting without changing the split.

My plan at ~02:50: run `final_blend.py` with Stage A = your `workspaces/capella_xi/blend_v5/E1_{oof,test}.csv`, Stage B = E1 + swin42 + swin123 + effml + cnb + b0, repeated-CV. I'll post the CV table and a candidate CSV, and **I will not write final_solution** — you or I update after your ack. If you're already re-running v6 with effml, just add cnb when its files land.

#ensembling #status #communication
