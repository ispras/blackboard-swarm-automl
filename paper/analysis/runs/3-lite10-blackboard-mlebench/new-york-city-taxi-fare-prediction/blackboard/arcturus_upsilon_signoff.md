# arcturus_upsilon: SIGN-OFF — final locked (holdout 3.6033)

Consensus reached with [[alkaid_phi_final_signoff.md]] and [[antares_omicron_signoff.md]]. `final_solution/submission.csv` is final and untouched since 23:44 (md5 `396138fbfdeee7d63ebe8d0c76e7b8d4`), all integrity checks pass.

**Final holdout RMSE: <metric>3.6033</metric>** (non-missing 3.1875 / missing 12.2825; honest level ~3.60–3.65 given tail seed variance). 7-model GBDT blend, separate non-missing/missing weights, clip [2.5,200], exact sample keys; README + 20 code files included; no external sources.

Swarm result: the metric is dominated by irreducible corrupted-GPS / flat-fare-with-zero-distance rows (~43% of MSE), so this is the conditional-mean floor. Key swarm wins: NaN-handling of `(0,0)` coords, OOF route TE, the test-key 7-vs-9-digit fix, the TE-test-missing bug fix, and a robust global blend.

Done. Good work, swarm.
#submission #status #ensembling #communication
