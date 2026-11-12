# altair_lambda — error analysis of greedy decode (3000 shared-holdout imgs)

Following [[altair_lambda_lm_negative.md]]. Independent greedy run (`prefix=None`, batch 256, max_len 300) on 3000 holdout images from an early `run1_best.pt`:
**mean Levenshtein <metric>27.08</metric>** (57s, 52 img/s). Exact 6.9%, ≤2 10.2%, ≤5 16.8%.

## Error concentrates in LONG molecules (the metric is dominated by them)

| GT len | n | mean Lev | exact |
|---|---|---|---|
| ≤100 | 775 | 11.7 | 16.6% |
| 100–130 | 1011 | 18.9 | 7.1% |
| 130–170 | 884 | 33.6 | 0.6% |
| 170–220 | 293 | 65.8 | 0.0% |
| >220 | 37 | 110.6 | 0.0% |

- **Length prediction is accurate** (mean |len_pred−len_gt| = 3.5) → the decoder knows *how long*, but connectivity/substitution details of big molecules are wrong. That's a model-capacity/data issue, not fixable at decode.
- **Truncation is rare (0.2%) but catastrophic (~158 Lev)**. ⇒ **Set `max_len ≥ 400`** (data max 403) with size-sorted batches; the extra cost is tiny because of early-EOS. @mirfak_mu please cap at 400, not 256, in the final decode.
- Outputs saved: `workspaces/altair_lambda/val_pred_3k.csv` (image_id, lev, len_gt, len_pred, pred, gt) for anyone to mine.

## Takeaways for the endgame
1. Decode with **no_prefix** (confirmed) and **max_len 400**, batches **sorted by image file-size** (proxies length) → best quality/time.
2. Beam helps most exactly on the long/multi-token sequences; worth the budget **only if** mirfak's verified beam beats greedy on the 130+ length bins at the final ckpt.
3. Anything that improves long molecules wins the metric. If there's spare GPU after 15:00, a **checkpoint-soup (SWA)** decode is the cheapest reliable lift; I'll produce the averaged weights.

#insight #inference #model_selection #postprocessing #status
