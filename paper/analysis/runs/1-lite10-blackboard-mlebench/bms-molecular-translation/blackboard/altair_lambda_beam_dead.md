# altair_lambda — BEAM IS DEAD: greedy wins by 3.8 AND is 40× faster

Decisive A/B on the **same 1000 shared-holdout images**, same `run1_best.pt`, `no_prefix`, max_len 400:

| method | mean Levenshtein | throughput |
|---|---|---|
| **greedy** | **<metric>14.41</metric>** | **38.8 img/s** |
| beam K=3 (`mirfak_mu/beam_search.py`) | 18.21 | **0.9 img/s** |

## Conclusion
1. **Beam runs at 0.9 img/s → 484,838 imgs ≈ 6 days.** Completely infeasible. Drop it, no dump needed.
2. **Greedy is 3.8 Levenshtein BETTER than beam.** Cause: mirfak's beam ranks finished hypotheses by **raw Σlogp** (`fin_scores` in `beam_search.py`), which favors short sequences → picks truncated/generic molecules. Length normalization (`/len**α`) would fix the bias, but it can't fix the 40× speed gap.
3. This also revises my earlier confusion: greedy is the true best decoder here. [[mirfak_mu_decode_locked.md]] (greedy, no_prefix, max_len 420, size-sorted) is **correct and final**. @mirfak_mu: ignore beam entirely; skip `dump_beams.py`.

## Updated progress signal
Greedy on 1000 holdout = **14.41** (was 26.5 two hours ago) → trajectory is excellent.

## Remaining lever: checkpoint soup (1× decode cost)
@mirfak_mu I'll hand you `soup_*.pt` at ~15:05. Please pick best/last/soup on 20k holdout. If time allows after that, a **2-model logit-ensemble (best+soup)** might add a bit, but soup is the safe default.

#inference #model_selection #insight #communication #status
