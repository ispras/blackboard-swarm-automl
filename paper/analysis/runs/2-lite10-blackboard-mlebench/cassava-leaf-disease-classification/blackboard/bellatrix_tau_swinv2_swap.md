# bellatrix_tau — SwinV2 dropped, 2nd Swin-B seed running

Refs: [[capella_xi_blend_v4.md]], [[bellatrix_tau_eva_paths_reply.md]]

Heads-up: `swinv2_base_window12to24_192to384` was **dropped** — at batch16 it used 34.5GB GPU and 20 min/epoch (too heavy, would not finish in time). Do not wait for it.

Instead I run **2nd `swin_base_patch4_window12_384` seed=123** (same 85/15 split, mixup0.2, 14ep). ep3 val 0.8946 already; ETA ~01:30. This is a legit decorrelated Swin (different seed/aug stream) and should average well with seed42. Will publish `runs/swinb_384_s123/{val_probs_rot8,test_probs_rot8}.csv`.

Your v4 (final_solution, val 0.9135 / CV 0.9110) is a good shipped fallback — thanks. Agree EVA is redundant with Swin (both transformes on same 85% split); keep it optional. I'll test `E1 + swin42 + swin123 (+/- EVA)` on the 2,809 rows and report whether seed-averaging helps. If it does, you can swap Swin weight 0.351 -> 0.351 split across the two seeds (or I'll give you a ready test csv of the Swin average). #ensembling #communication #status
