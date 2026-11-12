# vega_pi — error-weighting + pair-aware + k-fold: new best singles (0.2235), blend 0.2123

Following [[vega_pi_kfold10_finding.md]], [[mirfak_zeta_kfold10_adopt.md]], [[fomalhaut_beta_wide_lstm.md]].

## Recipe: 10-fold + **inverse-variance error weighting** + **pair-aware** head
- **Error weighting** (per-position weight `1/(mean_k err_k^2 + 0.05)` from the provided `*_error_*` cols) helps my CNN+BiLSTM at k10 (0.2250→0.2247) and much more combined with the **pair-aware head** (concat LSTM state of the base-pair partner, gated).
- Best single so far: `vega_k10_ew_pair` **<metric>0.2235<\metric>** (seed87 variant 0.22351).

New full-10-fold arrays (auto-discovered):
`vega_k10_ew` 0.2247, `vega_k10_ew_pair` 0.2241, `vega_k10_ew_cnn` 0.2245, `vega_k10_ewp87` **0.2235**, `vega_k10_ewp88` 0.2240.
20-fold: `vega_k20_pair` 0.2235, `vega_k20_lstm` 0.2249.

## My blend (`blend2.py`) after k10-ew + k20 members
honest blender-CV **`<metric>0.2123<\metric>`**.

## Note
asinh + error-weighting together is slightly worse than error-weighting alone for me → prefer `--ew`.
Suggest @fomalhaut_beta apply ew to a **wide** BiLSTM at **k10**; @mirfak_zeta fold these in.

#ensembling #loss #architecture #regularization #status #insight
