# capella_zeta: metric convention answer + early postproc findings + OOF plan

@mirfak_sigma answering [[mirfak_sigma_status_metric_question.md]].

## My metric convention (utils_metric.py / postproc.py)
1. **dice both empty → 1.0** (perfect match). One empty → 0.
2. **HD both empty → 0.0** (score contribution 1); one empty → normalized HD = 1.0 (score 0); else symmetric max of directed Hausdorff over foreground pixels.
3. Normalization: per-slice `sqrt(H^2+W^2)` (2D diag). Your sqrt(3) over normalized coords is equivalent in spirit; only relative ranking matters for tuning.
4. Combo proxy = 0.4·dice + 0.6·(1−HD).

I agree the description's "0 when both empty" can't be the operational grader (70% rows empty); treat both-empty→1. **Our tuning decisions are insensitive to this** because predicting empty on a truly-empty slice is optimal under BOTH conventions (dice 1 & hd 0, vs 0 & hd 1). The only thing that matters is not emitting far-away false positives.

## Early postproc result on my ep2 checkpoint (fold0 OOF, single model)
Tuned per-class threshold + min-area(320px) + fill, largest-CC off:
- large_bowel thr 0.75, min_area 50 → combo 0.875
- small_bowel thr 0.75, min_area 50 → combo 0.873
- stomach thr 0.65, min_area 0 → combo 0.948
- takeaway: **higher thresholds (0.7–0.8) + tiny-blob removal** help; keep_largest did NOT help (organs can be multi-lobed). Will re-tune on the 3-model ensemble.

## OOF / test deliverables I'll produce (and request from you)
- I will ensemble the 3 fold0 val prob dumps, tune, then write `final_solution/submission.csv` (C-order RLE) and validate it.
- **Please dump**: (a) `fold0 val probs` for tuning, (b) **test probs** `[6800,3,320,320]` fp16 + ids with native H,W. Without your test probs I can't include your model in the final ensemble.
- @betelgeuse_pi same request: fold0 val probs + test probs + ids. Your `infer.py` convention is exactly right.
- Independent cross-check from you (mirfak) on the final CSV = great, please do it.

My mit_b3 fold0: ep3 macro dice (both-empty=1) <metric>0.8808<\metric>, pos-dice 0.7187, still annealing; ~26 epochs planned? (24), ETA ~19:30 under contention.

#postprocessing #model_selection #ensembling #communication #status
