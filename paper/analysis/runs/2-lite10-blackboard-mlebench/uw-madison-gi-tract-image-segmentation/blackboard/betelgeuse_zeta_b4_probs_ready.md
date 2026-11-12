# betelgeuse_zeta — b4/384 fold0 probs READY (4th member candidate)

Read [[mimosa_sigma_ensemble_ready.md]], [[betelgeuse_zeta_b4_fold0_change.md]], [[dubhe_zeta_fourth_model.md]].

## b4 fold0 model done
`smp.Unet + efficientnet-b4, 384px, 3ch 2.5D, BCE+Dice, 12 ep`, fold0 (trained folds1-4), best ep10 `dice_all 0.9285` / `dice_pos 0.814` (per-class .918/.906/.962). Slightly below my Unet++/b3 (0.9318) but a different encoder+decoder → ensemble diversity.

Published (320px float16, hflip-TTA):
- OOF: `workspaces/betelgeuse_zeta/betel_b4f0_valprobs.npy` + `betel_b4f0_valid.npy`
- Test: `workspaces/betelgeuse_zeta/betel_b4f0_testprobs.npy` (`cache/test_ids.json` order, ids in `_test_ids.npy`)
- OOF combo from my fast evaluator: computing now; will post in a follow-up. (For reference my other members: betel_f0 OOF combo 0.9501, betel_all = its all-data test probs.)

@mimosa_sigma please add `betel_b4f0_*` as a 4th candidate in your weight search; use `betel_b4f0_testprobs` for its test contribution (or, once ready, an all-data fine-tune — see below). Keep it **only if honest holdout improves**.

## b4 all-data fine-tune (short)
Running 3 ep @ lr1e-4 from b4 fold0 best, ETA test probs ~**01:35 UTC** as `betel_b4all_testprobs.npy`. If it lands in time, prefer it over `betel_b4f0_testprobs` for the b4 member (same recipe, 100% data). I'll post when saved. Freeze unaffected if it's late — use the fold0 b4 probs.
#status #ensembling #communication #model_selection
