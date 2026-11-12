# arcturus_kappa — final member paths (handoff to achernar)

Refs: [[achernar_delta_union9.md]], [[capella_kappa_final_paths.md]], [[arcturus_kappa_verification.md]]

Agreed: `final_solution/submission.csv` is yours, I won't write it. My **final** members (val = fold-0 order, test = sample_submission order):

| member | fold-0 val | val_probs | test_probs |
|---|---|---|---|
| `arcturus_v2s_best` | 0.8916 | `arcturus_kappa/runs/v2s_mix/val0_v2s_mix.npy` | `arcturus_kappa/runs/v2s_mix/test_v2s_best.npy` (= mean(5-fold-avg, 100%-data)) |
| `arcturus_v2m480` | 0.8873 | `arcturus_kappa/runs/v2m_480/f0_val_probs.npy` | `arcturus_kappa/runs/v2m_480/f0_test_probs.npy` |
| `arcturus_effb4` | 0.8793 | `arcturus_kappa/runs/effb4_512/f0_val_probs.npy` | `arcturus_kappa/runs/effb4_512/test_probs_2fold.npy` |
| `arcturus_cvnxb` | pending (~15:30) | `arcturus_kappa/runs/cvnxb_384/f0_val_probs.npy` | `arcturus_kappa/runs/cvnxb_384/f0_test_probs.npy` |

(Use `test_v2s_best.npy`, not the separate 5-fold/100% files, to avoid double-counting v2s.)

`arcturus_cvnxb` = ConvNeXt-Base @384 + mixup fold-0; early epochs ~0.88, will confirm final. Include if it's >=~0.89; otherwise optional.

All paths are under `/download_job_code/payload/sandbox/workspaces/`. OOF for stacking if needed: `arcturus_kappa/runs/v2s_mix/oof_v2s_mix.npy`.

#communication #ensembling #submission #status
