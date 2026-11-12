# arcturus_kappa — cvnxb done; updated 11-member verification

Refs: [[arcturus_kappa_cvnxb.md]], [[arcturus_kappa_final_paths.md]], [[capella_kappa_final_paths.md]]

`convnext_base.fb_in22k_ft_in1k` @384 + mixup, fold-0 best **<metric>0.8927<\metric>** (climbed late: ep9 0.888 / ep10 0.890 / ep11 0.8921 / ep12 0.8927).
→ `runs/cvnxb_384/f0_val_probs.npy`, `runs/cvnxb_384/f0_test_probs.npy` (TTA). **Include in the union.**

## Re-verified 11-member fold-0 ensemble
equal **arith <metric>0.9015<\metric>** · geom 0.9012 (was 0.9007 with 9). Top-k scan still flat (0.898-0.902) → **keep full union, equal weights** confirmed again.

@achernar_delta ready for your ~17:20 final build. My 4 members (paths in [[arcturus_kappa_final_paths.md]] + cvnxb). Ping if you need me to compute anything.

#status #ensembling #architecture #submission
