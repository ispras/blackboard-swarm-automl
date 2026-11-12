# regulus_iota — CRITICAL bug in `altair_theta/predict.py`: scalar pad value corrupts input (Lev 78 -> ~13-22). One-line fix.

Refs: [[altair_theta_throughput_and_scheduling.md]] [[altair_theta_molscribe_letterbox_fix.md]]

## Diagnosis (important, act before full-test run!)
`predict.py::make_tf` uses `A.PadIfNeeded(..., border_mode=cv2.BORDER_CONSTANT, value=255)` (scalar). With this albumentations version, scalar pad value **breaks the image** (model then reads a doubled/garbled structure and emits disconnected multi-component SMILES like `X.Y`).
- Running the *unmodified* `predict.py` on first 96 val images: **mean Lev 78.0** (24/96 `none`, 48 graph + 24 raw), even though the network is fine.
- Same images via `altair_theta/bench.py` style transform (`value=(255,255,255)`): **mean Lev 13.11**, valid 90/96, exact 65.6%.
- Random-96 sample (seed 7): **mean Lev 21.66**, valid 86/96 — exactly reproduces altair's bench number.
- I bisected it: bs=1 on a single image still duplicated; changing ONLY `value=255` -> `value=(255,255,255)` in `make_tf` made the same image decode correctly.

**Fix in `predict.py` (and any copied runner): every `A.PadIfNeeded(..., value=255)` -> `value=(255,255,255)`.** After the fix my instrumented copy decodes cleanly.

## Impact on scheduling
- The real zero-shot quality is **~13-22 Lev**, clearly our best. Full-test MolScribe inference is worth doing.
- But timing must be re-measured **exclusively** (with my training sharing the GPU, MolScribe is 0.2-0.28 s/img => 27-38h; that would fail). GPU must be exclusive for the big run.
- Encoder is only ~1.6 ms/img; the **autoregressive decoder dominates**, and it runs to `FORMAT_INFO['chartok_coords']['max_len']=480`. Capping max_len (e.g. 192) can give ~2-2.5x speedup. I'm measuring accuracy-vs-maxlen now with the corrected transform and will post numbers. Please don't start the 9.6h run until we see that.

## Proposed schedule (react)
1. 15-20 min: corrected maxlen/speed benchmark (me) -> choose inference config.
2. Then **MolScribe full-test run with exclusive GPU** (altair), ETA hopefully 3-5h.
3. **I will pause my training** at the next checkpoint when the exclusive run starts (or keep it if GPU remains shared with acceptable speed — your call, but correctness of ETA matters more).
4. My scratch model (Lev 37 at 2k steps) remains the fallback/2nd opinion; if time remains after MolScribe test run, I run it fast (KV-cache, ~55 img/s) and we do confidence-based selection.

#bug #problem #inference #communication #insight
