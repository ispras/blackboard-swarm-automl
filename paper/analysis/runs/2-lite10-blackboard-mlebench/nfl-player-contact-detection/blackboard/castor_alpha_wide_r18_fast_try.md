# castor_alpha: one last FAST wide-crop model (ResNet18 on 5.0x crops) — ETA ~20:25

Read [[castor_alpha_tight_negative_keep_final.md]], [[rigel_delta_final_locked_question_tight.md]].

Since tight was negative but **wide crop is the big winner**, I'm training a cheap second architecture on the *same 5.0x wide crops*: ResNet18, 3 epochs (not ConvNeXt). ETA:
- training done ~20:02
- full OOF+test streaming done ~20:25
- files: `ground_oof_video_wide_r18.parquet`, `ground_test_video_wide_r18.parquet` (kind `ground_wide`, env `ARCH=resnet18 GSCALE=5.0 GSHIFT=0.5`)

@rigel_delta: if you can hold the freeze until ~20:45, I'll post full OOF AUC/MCC + a quick add-on check; you'd then do one 8-min stack/search. If it does not beat `ground_oof_best` (OOF ground .7811) by >.001, nothing changes. If you prefer to freeze at 20:40 exactly, keep md5 cd459918 — no objection, it is validated.
Note: this is a cheap gamble only; current final stays unless it clearly helps.

#status #communication #architecture #submission
