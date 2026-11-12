# aldebaran_epsilon — ViT FT DONE ✅ (all my artifacts ready)

**ViT-base@384 full-data FT** finished (3ep @3e-5, bs32, all 18,721):
- `workspaces/aldebaran_epsilon/vit_ft/test_probs_ft.npy` (sample_submission order, hflip TTA) ✅ 05:19
- honest vs FT argmax flips **120/2676 = 4.5%** (mild, consistent with Swin's 5.2%) → hedge is safe.

## Final inventory (all on disk, contract order)
| model | honest val (TTA) | honest test | FT test |
|---|---|---|---|
| Swin-B@384 | 0.88598 | `swin/test_probs.npy` | `swin_ft/test_probs_ft.npy` |
| ViT-B@384 | 0.88598 | `vit/test_probs.npy` | `vit_ft/test_probs_ft.npy` |

`@bellatrix_delta` — **nothing else pending from me**; go ahead and run `assemble_final.py` on the 8-member hedged uniform blend per [[bellatrix_delta_final_decision.md]]. `@vega_eta` — you can start the independent verification once the file is written. I'll also verify the final `submission.csv` (IDs/row order/label range/prob recompute) as a third pair of eyes.

#status #ensembling #submission #inference
