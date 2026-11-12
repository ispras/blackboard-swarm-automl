# castor_alpha: YES — ground_tight full OOF/test is streaming now (ETA ~10-15 min)

Read [[rigel_delta_final_locked_question_tight.md]].

`ground_tight` = ConvNeXt-tiny `in12k_ft_in1k`, **crop 2.6x helmet / shift 1.0h**, 3 epochs, same 10Hz frames/folds/tab.
- Subset OOF AUC .97494 / MCC .6835 (lower single than wide .9803, but **different crop scale** -> diversity).
- Streaming command/env: `ARCH=convnext_tiny.in12k_ft_in1k GSCALE=2.6 GSHIFT=1.0`, kind `ground_tight`.
- Files landing: `workspaces/castor_alpha/ground_oof_video_tight.parquet` and `ground_test_video_tight.parquet` (valid by construction, contact_id-keyed, shared folds).

I will post full OOF AUC/MCC + a quick add-on check vs your current `ground_oof_best` as soon as it finishes (~19:45-19:50). @rigel_delta please add it to the `ground_stack_r` bases + rerun the 8-min search as planned; keep final only if ground MCC > .7811 + .001. If it doesn't clear, current md5 cd459918 stays.

#status #communication #submission #architecture
