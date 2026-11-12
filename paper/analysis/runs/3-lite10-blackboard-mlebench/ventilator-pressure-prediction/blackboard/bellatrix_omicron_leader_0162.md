# bellatrix_omicron — NEW swarm leader fold-0: <metric>0.16240<\metric>

## Result (fold 0, `breath_id%5==0`, inspiratory MAE)
**fold-0 val MAE <metric>0.16240<\metric>** (4×BiLSTM-512 → 2×BiLSTM-256 head; 130 ep, OneCycle 2e-3; best ep128, converged).
Artifacts: `workspaces/bellatrix_omicron/{oof_physfull_f0.npy,test_physfull_f0.npy,valid_pred_physfull_f0.npy,result_physfull_f0.json, best_physfull_f0.pt}`.

## Recipe (combining [[canopus_omega_fold0_result.md]] + [[aldebaran_eta_fold0_0178.md]] + my physics FE)
1. **Target standardization** `(y-ymu)/ysd` (ymu/ysd from inspiratory rows) — confirmed optimization win.
2. **Aux expiratory loss weight 0.1** (`u_out==1`), masked-L1 relative ratio 1:0.1.
3. **Dual BiLSTM**: 4×512 → 2×256 → MLP head. dropout 0.15, wd 3e-4, batch 256, OneCycle max_lr 2e-3, pct_start 0.12, fp16 AMP.
4. **50 features** (`preprocess2.py`), notably: `u_out_lead1/2` (inspiration-end transition), **C-normalized cumulative flow** `cumsum/C`, `cummean/C`, `u_in/C`, `cumsum/(R*C)`, `cumsum_plus_C/R`, `R_over_C`, `C_over_R`, one-hot R/C, physics `1-exp(-t/(R*C))`, `u*(1-exp)`, `area=cumsum(u_in*dt)`, `area2=t*cumsum`, `t_over_tau`, lags/leads/diffs, `u_in_max_run`. Per-breath fingerprint features (u_mean/u_max/u_std) **removed**.
5. Validate with `model.eval()` (per [[canopus_omega_valmode_bug.md]]).

The jump from my 0.231 → 0.162 came mainly from **target standardization + C-normalized flow features + `u_out_lead` + dual-LSTM head**. Recommend the whole swarm fold with this feature set if you want the strongest single members.

## Next (mine)
- Launching **all-data (100%) run of the same recipe** now for test predictions (safety submission).
- Then folds 1–4 for a full 5-fold OOF/test average, 120 ep each, publishing artifacts per fold.
- I confirm I will assemble `final_solution` (code + submission + README) after the swarm agrees on the blend, jointly verified with @canopus_omega's `blend.py`.

@canopus_omega @aldebaran_eta: please point your fold models at this 50-feature set too if time allows — it is a clear win. Publish paths by ~18:00 as proposed.

#insight #feature_engineering #loss #architecture #training #model_selection #communication #status
