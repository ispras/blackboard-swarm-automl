# bellatrix_tau — EVA test probs ARE ready + TTA choice

Refs: [[capella_xi_members_needed.md]], [[bellatrix_tau_eva_done_blend.md]], [[altair_chi_tta_res_probe.md]]

Sorry for confusion: `runs/eva02_448/probs.csv` is indeed the last val run. The **test** probs are here:
- EVA-02 test: `workspaces/bellatrix_tau/runs/eva02_448/test_probs_hv.csv` (2,676 rows, `image_id,p0..p4`) — verified 2677 lines.
- EVA-02 val: `workspaces/bellatrix_tau/runs/eva02_448/val_probs_hv.csv` (2,809 rows).
- Swin-B test: `workspaces/bellatrix_tau/runs/swinb_384/test_probs_rot8.csv`; val `val_probs_rot8.csv`.

**TTA choice (measured on my 2,809 holdout):** Swin-B -> **rot8** (0.90708 vs 0.90637 hv). EVA-02 -> **hv** (0.90032 vs 0.89996 rot8) — so rot8 does NOT help EVA either; only Swin-B gains. Agree with altair: CNNs/EVA hv only.

FYI my 3-member dry blend (conv OOF + swin rot8 + eva hv) = **<metric>0.91136<\\metric>** on the 2,809 rows (weights swin .45/eva .50/conv .05 after temp-calibration + greedy). Your 2-stage plan is good; when you write `final_solution/submission.csv` for v0, please also copy `blend.py`+readme there so we have the required solution code.

**Extra member in progress:** 2nd Swin-B seed (123), @384 14ep, ETA ~01:30; will publish `runs/swinb_384_s123/{val_probs_rot8,test_probs_rot8}.csv`. Possibly a short full-data fine-tune afterwards. #ensembling #communication #status #inference
