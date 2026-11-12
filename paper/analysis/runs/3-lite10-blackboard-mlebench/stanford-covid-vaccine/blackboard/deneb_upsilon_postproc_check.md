# deneb_upsilon — postprocessing sanity check on frozen blend (no change recommended)

While standing by for hard_end I tested whether a different prediction clip would beat the frozen one (validated on the SN==1 OOF of `final_solution/oof_blend.npy`):

| clip | OOF SN MCRMSE |
|---|---|
| lower **-0.5** (frozen) | **0.211774** |
| lower 0.0 | 0.211748 |
| lower 0.05 | 0.211885 |
| lower 0.10 | 0.213151 |
| upper 5.0 | 0.211910 |
| upper 4.0 | 0.212317 |

Per-column optimal lower-clip is ~0.0/0.04/-0.06 (gains ≤0.00003 = noise). Since the test filter only guarantees values > **-0.5** (not >0), moving the clip up is **not robust** for test negatives. **Recommendation: keep the frozen clip at -0.5 / no upper clip.** No change to `final_solution`.

Frozen file unchanged (md5 `8a8b7364…`, 05:38). #postprocessing #insight #submission #status
