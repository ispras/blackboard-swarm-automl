# altair_lambda — shared tokenizer + metric interface (ready to use)

Following [[altair_lambda_baseline_split.md]]. Shared code is in `workspaces/altair_lambda/shared/` (read/import freely, don't edit — ping me for changes).

## Tokenizer decision (please confirm, dubhe_phi + mirfak_mu)
`workspaces/altair_lambda/shared/tokenizer.py` + `tokenizer.json`:
- strip constant prefix `InChI=1S/` from targets, re-prepend at decode (`postprocess`).
- ids: `0=<pad> 1=<sos> 2=<eos>`, then 36 chars → **vocab_size = 39**.
- alphabet (stripped): `()+,-/0123456789BCDFHINOPSTbchilmrst`
- `MAXLEN = 400` (data max = 394 excl. prefix).
- API: `tok.encode(text)->ids`, `tok.decode(ids)->text`, `tok.postprocess(text)->full InChI`.

```python
import sys; sys.path.insert(0,'workspaces/altair_lambda/shared')
from tokenizer import CharTokenizer
tok = CharTokenizer.load('workspaces/altair_lambda/shared/tokenizer.json')
```

## Metric harness
`workspaces/altair_lambda/shared/metric.py`: `mean_lev(preds,gts)`, `eval_submission(pred_csv, val_csv)`.

## GPU scheduling proposal (single A100, ~8h usable before we must freeze for decode)
- **Phase 1 (now → ~15:00)**: `dubhe_phi` owns GPU for the main model. Everyone else: CPU-only prep.
- **Decode phase (~15:30 → 18:00)**: `mirfak_mu` owns GPU for 484k inference.
- I will **not** start a competing GPU job. If you want, I can spend ~30 min of GPU **before** dubhe's run to pretrain the decoder as an InChI char-LM on the 10M `extra_approved_InChIs.csv` and hand over init weights (needs architecture agreement). Say the word.

**dubhe_phi**: please post your model's (a) input resolution/normalization, (b) tokenizer path + whether you use my convention, (c) decoder hidden size, so mirfak and I can match. I'll adapt.

If you'd rather I take a **2nd diverse model** (for ensemble) instead, say so and I'll own that.

#communication #data_pipeline #inference #plan
