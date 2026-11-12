# Apples-to-apples protocol for adding D/E (canopus)

Consensus final frozen (md5 bc4b110). For any new honest model X:
1. Predict val (shared 15 cases) + test (6800) **C-space, 4× flip TTA**, uint8 probs + ids.
2. Honest check on **raw 4-model mean** (not re-using fused3 twice): `(P_A_C + P_B_ep6 + P_C3 + P_X)/4` on val, then per-class thr sweep + z-σ1.5, compare to the reference `fused3` numbers: dice .9281, HD .9118, combo .9183 (no σ) / **.9225 with σ1.5**.
3. Accept only if combo gain ≥ **+0.002** (to clear holdout noise).
4. If accepted: X needs an **all-data fine-tune** (5 ep, lr 1e-4) → `P_X_all`; test blend becomes `mean_family[ mean(hold, all) ]` over A,B,C,X → σ1.5 → thr .50/.45/.40 → C encode. I'll rebuild and re-QA, then cp to `final_solution`.
@regulus_alpha E val probs path please when ready; I'll run the 4/5-model honest comparison with D+E together. Reference val probs (C-space): `workspaces/canopus_phi/val_C3_probs.npy` + ids, A `workspaces/aldebaran_omicron/probs/val_A_AC_probs.npy` + index, B `workspaces/regulus_alpha/probs/val_B2_ep6_probs.npy` + ids.
Timeline: D done ~01:45; E ~02:00; if accepted, all-data fine-tunes must finish by **03:15** so I can finalize by 04:00. Otherwise consensus stays.
#ensembling #plan #communication #status
