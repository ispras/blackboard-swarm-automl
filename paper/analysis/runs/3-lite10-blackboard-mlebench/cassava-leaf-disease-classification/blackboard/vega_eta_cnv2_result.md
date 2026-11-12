# vega_eta — ConvNeXtV2@384 DONE (honest) + FT running; 3rd model artifact paths

**Phase-2b `convnextv2_base.fcmae_ft_in22k_in1k_384` @384** (FCMAE-pretrained, distinct from supervised ConvNeXt) — 12 ep, fold!=0 train / fold0 val:
- best fold0 **no-TTA <metric>0.88945</metric>** (ep7). Trajectory .867 .877 .882 .880 .882 .876 .884 .889 .884 .888 .884 .885.
- Artifacts (contract): `workspaces/vega_eta/runs/cnv2b384_f0_s42/val_probs.npy` (fold0, no-TTA), `test_probs.npy` (hflip TTA, sample_submission order), `best.pth`; TTA-val + FT-all (`full/test_probs.npy`) finishing now (~04:30 / ~04:50).
@bellatrix_delta — I'll post the TTA val acc number when the quick recompute lands; use `val_probs_tta.npy` for honest weights like the others. Also: **pseudo-label experiment was negative** (EVA02 pseudo == honest 0.88865, dropped) — details [[vega_eta_pseudo_experiment.md]].

## My full member list for the final blend
| member | honest fold0 (TTA) | honest test | FT-all test |
|---|---|---|---|
| EVA02-B@448 | 0.88865 | `runs/eva02b448_f0_s42/test_probs.npy` | `runs/eva02b448_f0_s42/full/test_probs.npy` |
| CaFormer-b36@384 | 0.88892 | `runs/caformer384_f0_s42/test_probs.npy` | `runs/caformer384_f0_s42/full/test_probs.npy` |
| ConvNeXtV2-B@384 | ~0.889 (TTA pending) | `runs/cnv2b384_f0_s42/test_probs.npy` | `runs/cnv2b384_f0_s42/full/test_probs.npy` |
| (Swin, B5@512, B5@640, ConvNeXt-B, ViT) | — | yours | yours |

If time before ~05:30 I'll also try a validated **higher-res fine-tune of ConvNeXtV2 @512** (FixRes-style, 2 ep on folds!=0, checked on fold0) as an extra diverse member; will announce/drop if it doesn't beat 0.88945 honest. Otherwise I stop at 3 members and do the independent final verification at 06:30.

#status #training #architecture #ensembling #inference

---
**Update 04:20** — ConvNeXtV2 TTA-val done: fold0 **<metric>0.89159</metric>** (hflip TTA; no-TTA was 0.88945). That's now my *best single* member (EVA02 .88865, CaFormer .88892). `runs/cnv2b384_f0_s42/val_probs_tta.npy`. FT-all running.
Also started the validated **ConvNeXtV2@512** high-res fine-tune (2 ep on folds!=0, lr4e-5, bs16) → if fold0 TTA beats 0.89159 I'll add `runs/cnv2_512/{val_probs_tta.npy,test_probs.npy,val_acc.txt}` as an extra high-res member (ETA ~04:55). Otherwise dropped.
