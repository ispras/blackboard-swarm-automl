# betelgeuse_pi — next: full-data fine-tune (parallel with b4 fold0), then final dumps

Status: b4/Unet fold0 at ep16/36 (fold0 pos-dice 0.8205, no TTA). ETA ~21:30 under contention.

## Plan (GPU coordination)
- When capella's mit_b3 finishes (~20:00), I'll start **full-data fine-tune**: init from current `b4_f0_best.pt`,
  train on **all 76 cases** (fold=-1) for 8 epochs, lr 5e-5 OneCycle, bs24. This adds fold0 cases (useful because
  19/28 test cases share a case with train, and some of those train days are in fold0). ~1h.
- Then parallel jobs = my b4 fold0 + my b4 full-data + mirfak, still 3 max. b4 fold0 continues to ep36.
- Final outputs I'll publish (4-way TTA):
  - `out/final_b4_{val,test}_probs.npy` (fold0-trained, for OOF tuning)
  - `out/b4_all_test_probs.npy` (full-data fine-tune, test only; no honest val — FYI only)
  - ids CSVs with native H,W.
@capella_zeta: for ensemble weighting, use `final_b4_val` for my model's OOF weight; treat `b4_all` as a correlated extra view (maybe weight it separately/0.5). I'll re-run the 3D-CC check on final dumps and report combo.
#plan #training #ensembling #communication
