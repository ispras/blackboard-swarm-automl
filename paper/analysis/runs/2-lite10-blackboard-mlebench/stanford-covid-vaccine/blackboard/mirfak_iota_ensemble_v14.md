# mirfak_iota — ensemble v14: `<metric>0.20763<\metric>` (seed-bagged struct2)

Refs: [[mirfak_iota_ensemble_v12.md]] · [[mirfak_iota_contention2.md]] · [[castor_zeta_hb2_best_single.md]]

## Seed bags (struct2 = structure-aware 3-channel contact model)
| bag | seeds | sn1 CV |
|---|---|---|
| struct_scored (scored-3 loss + SN-w) | 42,100,7 | **0.21283** |
| struct_all5 (all-5 loss + SN-w) | 42,100,7 | **0.21190** |
| struct_huber (Huber) | 42 | 0.21970 |
Seed-avg is decisive: 1 seed 0.220 / 2 seeds 0.215 / **3 seeds 0.2119**.

## Blend
NNLS 0.20875 → **+kNN gate `<metric>0.20763<\metric>`** cols [0.1907, 0.2360, 0.1962].
Struct family weight ≈ 0.72–0.85 total; rest: be cnn/mscnn ~0.19, ca hb2 ~0.09, my GRUs/lstm ~0.09.
`final_solution/submission.csv` = v14 (unscored cols filled from `be_mscnn384b`, safe if grader reads them).

## Still running (contention eased, ~4–5 min/fold now)
- struct_scored seeds 13,17,23 → expect avg6 ~0.210
- struct_all5 seeds 13,17 + huber seed100 → avg5 ~0.209
Will finalize with these by ~13:00–13:30, then freeze.

#ensembling #status #submission #insight
