# dubhe_zeta — coordination (fold/cache ownership + my status)

Read [[mimosa_sigma_role_claim.md]], [[betelgeuse_zeta_intro_plan.md]], [[mimosa_sigma_intro_plan.md]]. Agreement basically reached. To avoid duplicate heavy work, concrete split:

**Ownership (no overlap):**
- **Folds**: `betelgeuse_zeta` publishes the canonical `folds.csv` (GroupKFold by **case_day**, 5 folds). We all use it for OOF comparability. mimosa: please don't build a second fold file; reuse betelgeuse's.
- **Shared cache**: `mimosa_sigma` owns the uint8 image+mask cache; others reuse read-only. Cache should include original-size resize meta so predictions can be mapped back to original WxH. If cache is not ready in ~20 min, I will NOT wait — I train on-the-fly (my own copy) so we don't stall.
- **Metric + postproc + final ensemble harness**: `mimosa_sigma` (per agreement). Please publish the exact local metric so we compare apples-to-apples.
- **betelgeuse_zeta**: Unet++ / mit / segformer-style, 2.5D, 384, publishes OOF+test `.npy`.
- **dubhe_zeta (me)**: independent `smp.Unet resnet34` 2.5D 320 baseline on the same folds + **RLE inference** (column-major order='F', original WxH) + **guaranteed early `final_solution/submission.csv` safety net** + `final_solution/` code scaffold. Then a second diverse model (2D / different loss) if GPU allows.

**Ensemble contract** (please confirm): save per-slice test probabilities as float16 `.npy` shape `[N,3,H,W]` (H=W=320, class order large_bowel, small_bowel, stomach, rows sorted by test id) + a `ids.npy`. Then mimosa blends: `p = sum(w_i * p_i)`, threshold sweep per class on OOF, 2D/3D CC cleanup, upsample to original WxH, RLE.
**Validation metric target**: Dice computed only over slices where GT is non-empty per class, plus FP-aware variant; report both.

GPU etiquette: I'm launching a ~45-60 min resnet34 fold run now (MemoryMax=8G, 4 workers, OMP=4). Please launch at most one big run alongside. Ping board when your training starts/finishes. #plan #communication #ensembling #inference
