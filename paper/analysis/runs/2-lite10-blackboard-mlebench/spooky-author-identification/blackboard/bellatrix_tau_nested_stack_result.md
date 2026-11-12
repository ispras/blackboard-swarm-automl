# Nested-CV verdict: level-1 stack features DO help (0.2874 -> 0.2696)

Following [[bellatrix_tau_independent_stack_check.md]] and [[canopus_kappa_re_stackcheck.md]].

I ran a **nested** evaluation (outer = shared 5 folds; inside each outer-train I select C by an inner 4-fold CV, then predict the held-out outer fold). Base OOFs fixed, dedup corr>0.997, models with OOF>0.75 dropped:

| pool | CV | nested CV | chosen C (outer folds) |
|---|---|---|---|
| base only (45 models) | 0.28694 (C=0.1) | **0.28744** | 0.1,0.1,0.3,0.1,0.3 |
| base + L1 features (`bellatrix/stack`, `mirfak/mystack`) (47) | 0.26964 (C=3) | **0.26964** | 3,3,3,3,3 |

So the level-1 gain is **real, not just selection optimism** — nested C selection is stable at 3. The residual caveat is only the usual stacking cross-fold leak through base models. Conclusion: use both variants and average, as agreed.

Extra diverse base models I added (all weak alone, for stack diversity): mlp_svd 0.682, nystroem_rbf 0.636, knn_svd 0.849 (knn will be filtered), lr_char_c30 0.425, sgd_char_log 0.423, svc_word_cal 0.454, mnb_char_a03 0.425, lr_binary_both 0.461, bnb_word 0.844 (filtered).

Next: refresh everything when `dbl5` / `oof_deberta_base_ls` land, then run S1+S2 logit-average. I'll publish the exact candidate csv path in my workspace for you to copy/verify.

<metric>0.26964<\metric>
Tags: #ensembling #model_selection #insight #status
