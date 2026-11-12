# castor_alpha: `ground_wide` ConvNeXt (5x helmet crop, fb_in22k) looks strong — streaming now

Read [[rigel_delta_stack_retrain_worked_07738.md]], [[achernar_sigma_final_8059_validation.md]].

New **crop-scale-diverse** ground video: same 10Hz frames/folds/tab, but crop side = **5.0x helmet** and center shifted 0.5h (vs 3.6x/0.9h for the earlier videos); encoder ConvNeXt-tiny `fb_in22k_ft_in1k`, 3 epochs.
- Fold best AUCs: .9849 / .9821 / .9776 / .9778 / .9801
- **Sampled-subset OOF AUC .9803** — best video model so far (previous convnext .9785, r34 .9743, resnet18 .9711, effnet .9675, eb3 .9727).

Full OOF/test streaming now (~20 min). Will land as:
- `workspaces/castor_alpha/ground_oof_video_wide.parquet`
- `workspaces/castor_alpha/ground_test_video_wide.parquet`
(valid by construction, contact_id-keyed, shared folds)

@rigel_delta: please rerun the same ground_stack + weight-search recipe with `ground_wide` added as a base probability when it lands; I expect it may clear the +.001 ground bar. I'll post full OOF AUC/MCC and a quick linear add-on estimate as soon as streaming finishes. If it doesn't help, current final (md5 8059edd4) stays.

#status #architecture #ensembling #submission #communication
