# spica_psi — fusion prototype WORKS (undertrained MLM + trigram → 4.33 on 150-val)

Re [[spica_psi_status_bugfix_eta.md]] [[antares_upsilon_trigram_results.md]] [[arcturus_kappa_detector_start.md]]

## Prototype result (small sample, genuine holdout, my own measurement)
Using the **undertrained ckpt_2000** MLM (gap-acc only 0.30) + antares trigram per-gap gain, candidate = **all interior gaps**, features `{mlm_p1, mlm_margin, mlm_top5mass, mlm_rank, tri_gain, tri_rank, word lens, same_word, ptri, k/M}` → logistic P(gap correct) fit on 150 holdout sentences, evaluated on the other 150:

| decision | metric |
|---|---|
| no insert (val baseline) | 5.2333 |
| trigram-only best gap+word | 4.7600 |
| **fusion, thr 0.5, MLM word** | **<metric>4.3267</metric>** (apply 35%, loc-precision 0.887) |
| fusion thr 0.4 | 4.3467 (apply 43%, loc 0.831) |

Takeaway: fusing **MLM prob** with **trigram gain** clearly lifts location precision (0.51→0.89 on applied subset) and MLM's word beats trigram's word at the fused gap. With the *final* MLM (+ arcturus' calibrated gap head) this should improve further. Caveat: 150-sentence val, noisy.

## Shared artifact (to save arcturus/antares recompute)
I already computed **per-gap best trigram gain for all 30k holdout sentences** using antares' `TrigramLM`:
`workspaces/spica_psi/tri_holdout_allgaps.pkl` → list[30000] of dict `{gap_k: (word, gain, logscore)}` (same convention, gap before token k). Cite [[antares_upsilon_trigram_results.md]].

## Updated ETA
GPU contention (arcturus 13.8GB + me 44.5GB, fine) slowed my FT to ~2.2 it/s → finish **~11:00**. Then I'll immediately produce the **30k holdout MLM npz**, then the **test MLM npz** (per-gap top-5). @arcturus_kappa you can keep training; I'll run inference alongside (my FT memory frees up first). If you want, initialize a second run from my **final** checkpoint later — it will be better than ckpt_2000.

## Fusion code
`workspaces/spica_psi/fuse.py` (logistic + expected-word selection). I'll publish holdout fusion numbers + the exact decision rule so antares can put it into the submission writer.

#insight #ensembling #model_selection #status #communication
