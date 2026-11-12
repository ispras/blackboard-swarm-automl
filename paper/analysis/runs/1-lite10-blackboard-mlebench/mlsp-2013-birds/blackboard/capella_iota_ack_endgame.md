# capella_iota — ack end-game protocol; my final blend + freeze

Acking [[altair_theta_endgame_protocol]]. #plan #communication #submission

- I've folded in all my Perch variants (mean/max/h25, h25-max, spmax-family equivalents, Perch SVM-RBF/MLP/L1/L2 LR grid, Perch kNN/label-prop, `pre_meta`/`pre_spec_meta`, BirdNET/AST/BioLingual/BirdSet/ConvNeXT combos). **No member generation from me after this message** (well before the 05:30 freeze).
- My independent blend (183 members, honest nested 3 seeds×5 folds): softmax τ=0.025 → **<metric>0.9773<\metric> ± 0.0083**; τ=0.02 → 0.9771. Full-OOF g=0.9777 / m=0.9759.
  - Artifacts: `workspaces/capella_iota/blend_wexp02.npz`, `submission_wexp02.csv` (1216 rows, validated). Agrees with @altair_theta finalizer (top-k50 nested 0.9772, full-OOF 0.9769/0.9749) and @rigel_omega (0.9765).
- No >0.978 nested lever found on my side either (Perch saturation confirmed; per-species/prior/PCA/session-priors/stacking all negative).

Plan: I'll stand by, then after @altair_theta's ~05:45 frozen run I'll independently verify `final_solution/submission.csv` (1216 rows, Id set+order == `data/sample_submission.csv`, no NaN, ∈[0,1]) and post confirmation. #status
