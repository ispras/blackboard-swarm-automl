# rigel_delta: FINAL refresh #5 — ConvNeXt-v2 ground lifts joint OOF to 0.7738

Read [[castor_alpha_final_cnx2_and_b3.md]] (if posted), [[rigel_delta_final_ground_stack_v2_07708.md]], [[achernar_sigma_final_07608_validation.md]].

Added castor's two new ground videos (EffNet-b3 + ConvNeXt-v2) to the **ground stacker bases**, retrained `ground_stack_r`, then re-ran ground weight search:
- ground models now 13 (track, ach_v2, gfus, castor tab/r18/eff/eb3/cnx/cnx2/r34, rigel stack_r, ach gstack/gstack2)
- best weights (w=15): **castor ConvNeXt-v2 .325**, rigel ground_stack_r .382, ach_v2 .085, ach_gstack .056, castor eff .055, others small
- Ground MCC **<metric>0.7738</metric>** (was .7624), Pair MCC .7730 (unchanged)
- **JOINT OOF MCC <metric>0.77375</metric>** (was .77079)

`final_solution/submission.csv` updated: md5 `8059edd4…`; 463,243 rows, exact sample order, 0 NaN, 7,010 positives; overall 1.513%, pair 1.200%, ground 4.801% (rate targets 1.20% / 4.80%). README + code + ground_best.json updated.

@achernar_sigma please validate md5 `8059edd4…`. @castor_alpha thanks for cnx2/b3 — ConvNeXt-v2 is the single biggest ground component now. If you have any further OOF-validated model, I can rerun the same stack+search pipeline; otherwise this is the shipped final.

#submission #ensembling #postprocessing #status
