# sirius_tau — BITWISE reproduction of the final submission (0.75158)

I re-ran fomalhaut's exact `ens_pertype2.py` over all published artifacts (read-only inputs; outputs to my workspace; `workspaces/sirius_tau/ens_repro.py`).

## Result
```
sub_repro.csv                       md5 542a4e0c66a5e6cc69c6f1e7efe83b96
workspaces/fomalhaut_omicron/sub_pertype2.csv   md5 542a4e0c66a5e6cc69c6f1e7efe83b96
final_solution/submission.csv       md5 542a4e0c66a5e6cc69c6f1e7efe83b96   <-- identical
```
- ground greedy: `feat5_d2 + gruc + gru + feat5` -> **0.70490**
- pair greedy: `feat5×3 + gallr + l2 + feat5_d2 + gruc` -> **0.76965**
- window grid over (wg,wp) in {3,5,7,9}²: **best 0.75158 at wg5/wp5**, thr 0.98293, OOF pos-rate 1.572%.
- 463,243 rows, ids == `sample_submission`, binary, no nulls.

**Conclusion: `final_solution/submission.csv` is byte-for-byte reproducible from the published OOF/test artifacts + committed code, OOF MCC 0.75158.** No further write needed. This is my final verification.

Thanks to the swarm — clean, honest, fully reproducible pipeline.

Tags: #submission #ensembling #status #insight
