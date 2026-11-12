# altair_lambda — LM fusion result (NEGATIVE) + greedy baseline on shared val

## 1) InChI n-gram LM built (works, but don't ship it)
Built a char 5-gram LM over **all 10M `extra_approved_InChIs` + 1.9M train** (1.52B tokens, stupid backoff).
`workspaces/altair_lambda/shared/lm_scorer.py` → `LMScorer.rescore(List[List[int]]) -> List[float]`.
Sanity: it scores true InChI **far above** corrupted ones (GT better 20/20; mean −145 vs −312).

## 2) N-best LM reranking = ≈0 gain — recommend DROPPING it
On `run1_best.pt`, 1200 shared-holdout images, top-5 beams (no_prefix, max_len 256):

| rerank | mean Levenshtein |
|---|---|
| model ranking (alpha=0) | 30.276 |
| + LM, alpha=0.1 (best) | **30.269** (−0.007) |
| + LM, alpha=0.2 / 0.4 | 30.31 / 30.34 (worse) |
| **oracle@5** (best candidate) | 30.023 |

→ **Both LM fusion and beam reranking have ~0.25 Levenshtein of headroom.** The decoder's own beam ranking is already near-optimal; the residual error is *semantic* (wrong molecule), not a ranking/typo problem. **No LM, no rescoring** ([[altair_lambda_rescore_iface.md]] withdrawn).
Caveat: my quick `beam_topk` was slower/worse than expected — see (3).

## 3) Greedy is fast and strong (and better than my beam test)
Same 1200 shared-holdout images, `run1_best.pt`, **greedy `prefix=None`**, batch 256, max_len 256:
- **mean Levenshtein <metric>26.52</metric>**, 20s (58.6 img/s) — at a checkpoint where dubhe's prefix-quickval was ~41.5.

⚠️ My top-5 beam gave 30.3 — **worse than greedy**, which means my beam selection is buggy, not that beam is bad. @mirfak_mu your bit-exact `beam_search.py` is the source of truth; please **A/B greedy vs your beam(K=3)** on a fixed 1200 subset with the *final* ckpt and pick the winner under the decode-time budget.

## 4) Decode throughput knob
Throughput is dominated by **early-EOS + batch length homogeneity**. Sorting test images by (predicted) length / file size and capping `max_len` matters a lot. mirfak already plans size-sorting ([[mirfak_mu_floor_update.md]]).

## 5) I'll prepare decode-time lifts for the endgame (no GPU now)
- **checkpoint soup / SWA** of last-N epoch ckpts,
- **TTA** (2 scales) + argmax by model score,
- independent final `mean_lev` verification.
Ready to run the moment training stops (~15:00).

#ensembling #inference #model_selection #insight #status
