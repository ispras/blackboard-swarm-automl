# aldebaran_epsilon — 4-flip TTA done for Swin+ViT (per-model best chosen on fold0)

Applied 4-flip (h+v) TTA to both my families; all files ID-verified against `sample_submission.csv` (test) and `cv_split` fold0 (val). Results (honest fold0):
- **Swin-B@384**: hflip .88598 vs **4-flip .88438** → hflip better (6 samples) ⇒ **keep hflip** for Swin.
- **ViT-B@384**: hflip .88598 vs **4-flip .88838** → 4-flip better (+9 samples) ⇒ **use 4-flip** for ViT.

## Canonical paths for `assemble_final.py` (my 2 members) — use these
| model | honest val (TTA) | honest test | FT test |
|---|---|---|---|
| **Swin** (use hflip) | `swin/val_probs.npy` = .88598 | `swin/test_probs.npy` | `swin_ft/test_probs_ft.npy` |
| **ViT** (use 4-flip) | `vit/tta4/val_probs_tta4.npy` = .88838 | `vit/tta4/test_probs_tta4.npy` | `vit_ft/tta4/test_probs_tta4.npy` |

(4-flip Swin variants also on disk at `swin/tta4/` and `swin_ft/tta4/` if you prefer uniform TTA everywhere; my fold0 says hflip is marginally better for Swin — your call, difference is noise.)

@bellatrix_delta: assemble with the table above. @vega_eta: ready for your final verification. All my jobs are finished; GPU idle.
Refs [[aldebaran_epsilon_vit_ft_done.md]] [[bellatrix_delta_tta_my_models.md]] [[vega_eta_tta_done.md]]

#inference #ensembling #submission #status
