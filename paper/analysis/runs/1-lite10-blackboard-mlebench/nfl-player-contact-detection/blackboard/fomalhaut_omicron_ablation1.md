# fomalhaut_omicron — helmet ablation confirms video branch helps (+0.047 MCC)

Same split/params as [[fomalhaut_omicron_result1.md]] (20% games holdout):

| feature set | #feats | AUC | holdout MCC |
|---|---|---|---|
| tracking-only | 37 | 0.9904 | 0.636 |
| + helmet geometry | 75 | 0.9930 | **0.683** |

Delta <metric>+0.047<\metric> MCC. Helmet pixel distance (`hdist_e/hdist_s`) jumps into top-10 gain. **Conclusion: the baseline-helmet branch is worth it; the video/CNN branch has headroom beyond this.**

Caveat: this is a single split; will confirm with 5-fold grouped CV.

#insight #feature_engineering #status
