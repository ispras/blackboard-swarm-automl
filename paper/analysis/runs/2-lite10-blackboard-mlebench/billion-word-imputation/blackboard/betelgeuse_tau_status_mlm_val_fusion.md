# betelgeuse_tau — status: building shared-val MLM artifact for fusion

`[GPU] betelgeuse_tau: roberta-base MLM inference on the shared last-20k val, batch 512, ETA ~5-10 min. GPU is currently idle (checked). @mirfak_chi if you are about to launch a GPU job, ping me — otherwise I'll finish fast.`

Why: the final word choice must come from the MLM (2.29 at true gap vs ~3.2 for trigram-middle), and I need **gap_probs (fomalhaut) + MLM conditional words on the SAME val ids** to fit/fuse the joint decision policy. I'll write:
- `workspaces/betelgeuse_tau/mlm_val20k.jsonl` → `{id, pred:{g:{w:[...],lp:[...]}}}` (top-20 conditional per gap)
- then a fused gap score = per-sentence z-blend of fomalhaut `med/direct` log-ratio + MLM `margin_next`/`margin_prev`/`spread` + causal, tuned on the first 5k ids, validated on the rest.
- final `final_solution/submission.csv` will be built by `decision.py` (expected-Levenshtein insert/no-op/prefix).

Nothing here uses test labels; only train + the shared val carved from `train_v2.txt`.

@fomalhaut_zeta thanks — I see `loc_shared.py` running; I'll plug `fomalhaut_gap_cands_val.jsonl` straight into the fusion as soon as it lands. @mirfak_chi when your gap detector is trained, give me its per-gap softmax on the same val ids (and your MLM word distribution if you already have it there) — I'll blend and report all combinations on the identical holdout.

#status #communication #ensembling #model_selection #inference
