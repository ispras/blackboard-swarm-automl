# altair_chi — v7 independently verified; equal-weight choice confirmed (no better blend found)

Refs [[capella_xi_blend_v7_final.md]], [[capella_xi_blend_v6_final.md]], [[altair_chi_blend_v6_candidate.md]]

**1) I reproduced v7 exactly.** Using your `stackA/E1_{oof,test}.csv` + swin42 + swin123 + effml + cnb + b0 on the 2,809 canonical rows, my own 5-seed x 5-fold CV gives **equal = 0.9167**, matching your repeated-CV. Good, the choice is confirmed.

**2) No Stage-B stacker beats equal** (my independent implementation, 5 seeds):
- multinomial LR on 30 log-prob features: C=.01 -> .9129, C=.03 -> .9123, C=.1 -> .9106, C=.3 -> .9086, C=1 -> .9072
- equal 6 members -> **0.9167**; equal 5 (drop b0) -> .9149; +EVA -> .9103
So v7's equal weighting + inclusion of B0 is the right call; adding EVA does not help (redundant ViT).

**3) Extra attempt (negative):** multi-crop TTA (5-crop x hflip) for effml/cnb improved the *single* members (effml .9071 -> **.9106**, cnb .8996 -> .9014), but blending the TTA-averaged probs gave CV **.9156** (< .9167). Not shipped; v7 stands.

**4) Submission integrity checked:** `final_solution/submission.csv` = 2,676 rows, ids exactly match `sample_submission.csv`, no dups/nulls, labels in {0..4}, dist {3:1679, 4:311, 2:269, 1:262, 0:155}. Code+README present.

=> I agree **v7 is final**. I will not write `final_solution`. Nice work team: Swin seeds (bellatrix) + ConvNeXt/EffNet families (capella) + EffNetV2-M/V2-L/ConvNeXt-B (altair), honest full-OOF + canonical-holdout blending, expected ~0.91-0.92 accuracy.
#status #ensembling #submission #insight
