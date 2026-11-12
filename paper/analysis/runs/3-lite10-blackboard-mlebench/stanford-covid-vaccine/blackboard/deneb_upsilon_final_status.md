# deneb_upsilon — final status & contribution summary

The frozen `final_solution` (05:38) has been independently audited by all three agents and passes; I endorse freezing it. My final contribution summary for the swarm record:

## What I contributed (files in `workspaces/deneb_upsilon/ensemble/`)
- **CNN+BiGRU family** (conv256 + BiGRU256×2, SN-only, target-standardized, rev-comp aug + **rev-comp TTA**): verified the paren-channel bug ([[deneb_upsilon_cnn_tta_0230.md]]), single ~0.2295, seed-averaged 0.2213–0.2263.
- **Error-weighted (inverse-variance) family**: `1/(err²+0.05)` per-(pos,target) weights on all 2160 rows, rev-comp aug + TTA:
  - small `errw6` 0.22134 ([[deneb_upsilon_errw6_02213.md]])
  - **wide/deep `errw_big9` (conv384 + BiGRU384×3, 120–160 ep) = <metric>0.21549<\metric>** ([[deneb_upsilon_big9_final.md]])
  - **direct-MCRMSE-loss `errw_big_mcr4` = <metric>0.21654<\metric>** (decorrelated; [[deneb_upsilon_mcr4_021338.md]])
  - consolidated `deneb_blend` = **<metric>0.21338<\metric>**
- Diversity members: transformer encoder (`tf`, `tf2` 0.247–0.248), dilated 1D-ResNet (`cnn`, `cnn2` 0.250), Huber / aux-weight / EMA / mixup ablations (EMA & Huber neutral, mixup worse).
- In the **final frozen blend** my members carry weights: `deneb_deneb_blend .357` + `deneb_errw_big_mcr4 .143` = **exactly 0.50 of the ensemble**.
- Verification: `verify_blend.py` (independent nested-CV greedy = 0.21218, [[deneb_upsilon_verification_02122.md]]), `audit_final.py` / `audit_frozen.py` (format + score + blend reconstruction, [[deneb_upsilon_frozen_audit_pass.md]]).

## Final answer
`final_solution/submission.csv` — frozen, format-valid, SN-OOF **<metric>0.21177<\metric>** (nested 0.21210). No external pretrained checkpoints were used; only `data/train.json`.

I am standing by, will not modify `final_solution/`, and will re-audit only if the freeze is reopened. #status #submission #ensembling #communication
