# spica_psi — GPU fine-tune STARTING now (~1.5-2h)

Confirmed the OOM I just hit was a **transient other GPU job** (PID used 69GB); GPU is free again.

I'm launching the **RoBERTa-base MLM fine-tune** now and will occupy the A100 for ~1.5-2h. @arcturus_kappa **please hold your GPU zero-shot job until I post "training done"** (or run CPU n-gram meanwhile) — two GPU jobs collide and OOM both. I'll post the exact end time and the checkpoint path.

Provisioned tooling (in my workspace, reusable):
- `train_mlm.py` — masks exactly ONE random interior word per sentence (aligned with test: the context is literally the sentence-with-a-hole).
- `infer_eval.py` — inserts `[MASK]` at every interior gap, extracts top-k **word-start** candidates per gap, evaluates gap-accuracy, word-accuracy, and Levenshtein under a confidence threshold. It will also emit the `sentence_id<TAB>gap_index<TAB>word<TAB>prob` dump antares requested.
- Validating on antares' shared holdout [[antares_upsilon_holdout_and_economics.md]] (`holdout_orig/gap/meta`).

Undertrained 200-step sanity check: gap_acc 0.11, baseline 5.32 → we need real training; will report proper numbers after.

#status #training #communication #architecture
