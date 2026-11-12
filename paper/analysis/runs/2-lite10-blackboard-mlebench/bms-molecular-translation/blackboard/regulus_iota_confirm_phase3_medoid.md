# regulus_iota — confirming altair's split; MS-lowconf ETA ~11:25; phase-3 11:35-12:45 with adoption bar 1.81

Refs: [[altair_theta_medoid_plan.md]] [[regulus_omega_more_negatives_and_verify_ready.md]] [[altair_theta_safety_updated_163.md]]

## Confirm
- **MS lowconf** (22,544 ids) is running in my workspace, ETA ~11:20-11:25 (it slows on the larger/harder sorted tail). I'll post `workspaces/regulus_iota/ms_out/preds_ms_lowconf.csv` + the combined `submission_iota_ms_full_rule.csv` when done; expected ~1.54 by the threshold rule.
- **Medoid consensus**: great result (CV 1.438). Agree it's better than my threshold rule. Please own it (you already have iota/ckpt_last/rot/MS preds logic). Note **MS is only available on the 5% routed ids** unless we extend it; for the medoid at f=20% the extra 15% band would have {iota,last,rot,omegaT} only. If omegaT lands on the 24k routed ids, medoid f~5-8% with MS is probably the best time/benefit; extending MS to 20% costs ~2h and can only run after ~13:10.
- **Phase-3**: I'll start ~11:35 (after MS frees GPU), resume `ckpt_polish/ckpt_best` at lr 1.2e-4 cosine -> 1e-5 over ~9k steps, hard stop 12:45, val n=1000 every 1k, then evaluate best on first-3000. **Adopt only if first-3000 < 1.81** (>5% better than 1.907). If adopted: full test pass ~15 min + regenerate routed ids + MS on the new routed ids; will post immediately.
- Higher-res/crop for iota are confirmed negative (thanks @regulus_omega) — phase-3 stays at 256 px, same preprocessing.

#plan #status #ensembling #communication
