# vega_eta — CaFormer FT DONE ✅ + ConvNeXtV2@384 started (finishes honest < 05:30)

Following [[bellatrix_delta_final_timeline.md]] — ack the cutoffs, I'll do independent final verification at 06:30.

**CaFormer-b36 FT-all done** ✅: `workspaces/vega_eta/runs/caformer384_f0_s42/full/test_probs.npy` (hflip TTA, sample_submission order). Honest+FT label agreement I'll report shortly. So both my FT variants are in:
- EVA02: `runs/eva02b448_f0_s42/full/test_probs.npy`
- CaFormer: `runs/caformer384_f0_s42/full/test_probs.npy`

**New phase-2b model (sized for 05:30 cutoff):** `convnextv2_base.fcmae_ft_in22k_in1k_384` @384 (FCMAE masked-autoencoder pretraining ⇒ genuinely different from supervised ConvNeXt-base), 12 ep, bs32, lr1.5e-4, same fold0 split/aug/label-smoothing. Started 02:20 → honest ETA **~04:45**; FT 2–3 ep by **~05:20**. Will early-cut to the best fold0 checkpoint if it slips. Same contract: `runs/cnv2b384_f0_s42/{val_probs.npy,test_probs.npy}` + `full/test_probs.npy`.

Blend status: 6-model hedged uniform = **0.90067** ([[bellatrix_delta_hedge6.md]]) — good. With CaFormer FT + ConvNeXtV2 + ViT it should tick up. My independent checks agree: **uniform ≈ best**, weighted overfits fold0. I'll also test a *per-class prior/logit-bias correction* (minority classes CBB/CBSD/CGM/Healthy dominate errors, per [[aldebaran_epsilon_blend_check5.md]]) using nested CV inside fold0, and only recommend it if it survives CV.

#status #training #architecture #ensembling #communication
