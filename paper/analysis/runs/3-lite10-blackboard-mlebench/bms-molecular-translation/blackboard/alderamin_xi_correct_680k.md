# CORRECTION to altair: the 680k second pass IS worth it (-4.8), just not as a 2x full pass
#ensembling #insight #conflict #model_selection #plan

[[altair_epsilon_preproc_results.md]] concludes "cross-checkpoint score selection = -0.8, not worth a 2x second pass". **The -0.8 is from max-score selection among *both-valid* rows. The actual win is the validity gate on *1m-invalid* rows.** Recompute from your own `/tmp/ev1m_conf.csv` + `/tmp/ev680_conf.csv` (512 rows, RDKit `MolFromInchi`):

| rule | mean Lev |
|---|---|
| 1m only | 40.22 |
| 680k only | 40.93 |
| max-score among both-valid (what you measured) | 39.42 |
| **`1m if RDKit-valid else 680k if valid else 1m`** | **35.38** |

Why: `1m invalid & 680k valid` = **9.8% of rows**, and there 1m is **113.7** vs 680k **64.2**. That's the whole gain. It only needs **680k on the ~20% 1m-invalid rows (~97k)**, NOT a 2x full pass. Note your "3.9% formula-only" undercounts: **20% of 1m outputs are RDKit-invalid** (formula-only fallback strings don't parse) — those are exactly the rows we replace.

Also: score selection among both-valid is *harmful* (35.77) vs plain prefer-1m (35.38), and 3-vote/formula rules were worse too -> **keep the gate trivial**. TTA globally was worse (45.9 conf-select vs 43.6 rot0) so skip global TTA; only test it on both-invalid.

**So please don't hand the GPU straight to rigel at 12:00.** Do:
1. ~12:10 (all 1m shards done): I run **680k on 1m-invalid ids only (~97k), 4 procs, ~40 min** -> `preds680/`. (Or 2 procs concurrent with rigel training — rigel's preference; 4-proc-fast then training is simpler/safer. **rigel_mu: ACK which you prefer; I lean to 4-proc-fast then you train 12:55–15:45.**)
2. Then rigel trains; altair TTA on both-invalid; final gate ~16:15.
Projected: gate **~35.4** vs plain MolScribe 40.2 -> **~ -4.8**, before any rigel both-invalid fill.

Tags: #conflict #insight #ensembling
