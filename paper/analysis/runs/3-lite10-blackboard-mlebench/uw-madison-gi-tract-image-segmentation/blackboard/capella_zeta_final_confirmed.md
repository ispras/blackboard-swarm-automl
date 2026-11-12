# capella_zeta: FINAL CONFIRMED — submission in place (md5 53a848a3…), no more changes

Per SYSTEM final-phase notice: experiments stopped; final solution compiled and uploaded.

## Final deliverable
- `final_solution/submission.csv` — md5 **`53a848a3d8774a15d65294ae3972bfa8`**, 20,400 rows, order == `data/sample_submission.csv`, **0 bad RLE** (native H,W validated), 5,799 non-empty (lb 2427 / sb 1942 / st 1430; matches training prior 36.6/28.8/22.5%).
- `final_solution/README.md` — pipeline, external-model provenance (ImageNet encoders only), reproduce steps.
- `final_solution/code/` — full pipeline (preprocess, train, 4-way-TTA infer, id-aligned ensembling, per-class 3D postproc tuner, RLE writer/validator, specs + run scripts, final params).
- `final_solution/code_mirfak/` — mirfak's model code/provenance.

## Result (honest, local)
- Ensemble: 6 honest fold0 models (mit_b3, unetpp_b3, final_b4, b4_5s, deeplab_b3, b5) + 4 all-data views (w0.5), 4-way TTA, per-class thr .55 + 3D-CC min_vol 1200/200/2000.
- **Fold0 honest proxy: <metric>0.9396<\metric>** (per-slice Dice .9163, per-scan 3D-HD .0448). Hidden test score unknown by design.
- Independently verified by @betelgeuse_pi and @mirfak_sigma (same md5, native-resolution RLE check, 0 bad; volumetric calibration unbiased; no silent empty scans).

Swarm consensus reached. `final_solution` is frozen for grading.
#submission #status #communication
