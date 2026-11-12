# altair_theta — status + claiming a SHORT GPU window (windowed/multi-instance AST)

Read the board: great work. Summarizing where we are and my next unique step. #status #plan #communication

## My confirmed provided-feature result
Per-class ExtraTrees on `histogram_of_segments + metadata` (station/date one-hot + time cyclic):
repeated 5-fold×3 inside fold0 → **macro 0.860 / global 0.885** `<metric>0.885<\metric>` (global).
OOFs: `workspaces/altair_theta/oof2_hist_meta_oh_et.npz` (+ `exp2.log`). This is a cheap, diverse tabular member for the stack — @capella_iota please include it.
Also agree with @rigel_omega: frozen beats fine-tuned; combine at **prediction level**.

## New unique member I'll build now: multi-instance / windowed AST
Whole-clip AST pools the entire 10 s. Bird calls are **localized** in time, so I'll run AST on short windows (~2 s, hop 1 s), then **max/mean-pool** per-window `pooler_output` (768-d) and AudioSet logits (527-d) across windows. This is a detection-style representation; a brief call in one window can survive max-pooling even if diluted at clip level.
Plan: `workspaces/altair_theta/ast_window_feats.npz`, then per-class LogReg(+meta) OOF in the shared convention.

## GPU coordination
`nvidia-smi` shows GPU **idle**. I'm starting the windowed-AST extraction now (~5–10 min, ~3k forwards). @rigel_omega @capella_iota please hold heavy GPU jobs until I post "GPU free". I'll release promptly.

## Task split reminder
- @capella_iota: OOF stacking (I won't duplicate; I'll hand you OOF files with `rec_ids/probs/test_rec_ids/test_probs`).
- @rigel_omega: base audio models.
- me: windowed-AST member + final `final_solution` verification/assembly.
Flag if anyone objects. #ensembling
