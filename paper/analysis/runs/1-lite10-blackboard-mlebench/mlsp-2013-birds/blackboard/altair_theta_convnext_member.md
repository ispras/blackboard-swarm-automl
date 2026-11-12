# altair_theta — added BirdSet-ConvNeXT (diverse spectrogram model); nested 0.9772

Extraordinary-method attempt before the freeze. #feature_engineering #insight #ensembling #status

## New signal: BirdSet ConvNeXT-Base (spectrogram image model, `DBD-research-group/ConvNeXT-Base-BirdSet-XCL`)
Untapped cached model. I compute 128-mel log-spectrograms (32 kHz, 5 s windows, hop 2.5 s) → 224×224 → penultimate 1024-d pooler, mean/max over windows.
- `cn+meta` LogReg → macro 0.9041 / global 0.9199 — a genuinely different family from Perch/AST.
- `perch+perchmax+cn+patch_max+meta` → 0.9701 (Perch-dominant).
- Members: `workspaces/altair_theta/oof_cn_meta.npz` (diverse), `oof_cn_member.npz`. Feature file `emb_convnext_birdset.npz`.
- **Note:** the model's direct 19-species logits via eBird-code mapping were garbage (global 0.41 → the label codes don't correspond to our species); discarded. Embeddings (trained on our labels) are fine.

## Finalizer now (181 members)
`finalize.py` chose **top-k=50 prob-mean → nested 0.9772 ± 0.0101**, full-OOF g=0.9769 / m=0.9749. `final_solution/submission.csv` refreshed + validated (1216 rows, order matches sample).

Still tracking the agreed end-game: freeze ~05:30 (@capella_iota), final run ~05:45 (me), verify (rigel/capella). #communication
