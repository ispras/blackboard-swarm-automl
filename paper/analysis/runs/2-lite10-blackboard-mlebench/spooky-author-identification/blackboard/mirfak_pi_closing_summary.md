# mirfak_pi — closing summary (final state verified)

## Final artifact
`final_solution/submission.csv` (written by canopus_kappa, 01:11) — **honest base-only 3-way stack**, CV **0.19863** (canopus) / **0.19839** (mirfak) / **0.19828 nested** (bellatrix). Verified now:
- 1,958 rows, `id,EAP,HPL,MWS`, ids == `data/test.csv` order, all finite, rows sum to 1
- identical to `solution/submission_reproduced.csv` (self-contained reproduction from `base_oof.npz`/`base_test.npz`)
- corr 0.9995 with my independent 69-model honest rebuild (argmax 99.4%)

## My contributions
1. **Transformer track**: `deberta_base_ls` 0.35472, `roberta_large_snap` 0.35690, `xlnet_large_snap` 0.37671, `deberta_base_aug` 0.35477.
2. **Best single model**: `dbl_aug_snap` = deberta-v3-large + **token-masking augmentation** (p=0.1) + label smoothing + per-fold snapshot averaging → **OOF 0.30676** (swarm best). Scripts in `final_solution/solution/teammate_mirfak_*`.
3. **Leak discovery**: caught double-stacking via level-1 files (`final_meta`, and later case-sensitive name `FINAL_honest` / `verify_allowlist`) that produced fake CVs of 0.165-0.194. Proposed the **lowercase substring filter + <0.25 individual-logLoss guard** now used for the final.
4. **Meta-model validation**: log+prob features; LR/HGB/XGB blend; nested verification that the nonlinear gain is real; showed per-model temperature, length features, MLP 4-way and prior correction do **not** help.

## Final honest numbers
- Level-1 best singles: dbl_aug 0.3068-0.3118, dbl5 0.3173, deberta-base 0.3547, roberta-large 0.3569, bert5 0.3658, xlnet 0.3767, TF-IDF linear 0.3737, BiLSTM 0.4293.
- Level-2: 0.60·LR(C=.05-.1) + 0.20·HGB + 0.20·XGB on `[log p, p]` → **<metric>0.19839<\metric>** (final honest estimate 0.1984-0.1987).

Swarm was efficient: 3 independent stack implementations agree within 0.0005. Final is frozen; no further writes from me.

Tags: #status #submission #ensembling #insight #communication
