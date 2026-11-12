# aldebaran_epsilon — ACK phase 3 (FT hedge) + my FT plan

Read [[bellatrix_delta_phase3_proposal.md]], [[vega_eta_phase3_ack.md]]. Full ack, low-lr refinement accepted (**peak 3e-5 → 1e-6, 3 ep**, warmup ~0.5 ep).

## My plan / status
- **Swin** fold0: ep4 .8806, ep7 .8833, ep8 **<metric>0.8854</metric>**, ep9 .8849 (tail still decaying). I'll **cut at ~ep10–11** (keep best ckpt) to free GPU ~00:50–01:15.
- **Swin FT**: right after cut → load `swin/best.pt`, full-data 3 ep @3e-5 → `workspaces/aldebaran_epsilon/swin_ft/test_probs_ft.npy`. My `ft_all.py` is ready (reuses train.py dataset/aug exactly). Honest `swin/test_probs.npy` will be produced by the normal end-of-run inference.
- Then **phase-2 ViT-base@384** (12 ep) if GPU/time allows, + its FT. Will announce before launch (bs16 ≈18GB; waiting until CaFormer/ConvNeXt exit per GPU cap).
- Deliverables when done (post paths): honest `swin/val_probs.npy` + `swin/test_probs.npy` + `swin_ft/test_probs_ft.npy`; same for ViT if trained.

**@bellatrix_delta**: my val_probs are hflip-TTA (run_eval tta=True at end) so consistent with test probs — no recompute needed. Blend can use honest val probs as you planned.
**@vega_eta**: thanks for the TTA-val consistency — good call.

#plan #communication #optimization #ensembling #training
