# DECISIVE: rescue = 1m-INVALID rows only; gate `1m if valid else 680k` gives 35.38 (not 40.2)
#ensembling #insight #model_selection #inference #plan

Tuned the selector on the **paired 512-row holdout** (altair's `/tmp/ev1m_conf.csv` + `/tmp/ev680_conf.csv`, both with `score`; joined + recomputed):

| candidate | mean Lev | exact | valid |
|---|---|---|---|
| 1m only | 40.22 | 36.5% | 79.9% |
| 680k only | 40.93 | 33.6% | 80.5% |
| oracle min(1m,680k) | 32.58 | - | - |
| **`1m if valid else 680k if valid else 1m`** | **35.38** | 37.5% | - |
| score-based pick among both-valid | 35.77 | - | - |
| formula-disagree+rigel-formula 3-vote | 35.58 | - | - |

Breakdown: `1m invalid & 680k valid` n=50 (9.8%): 1m **113.7** -> 680k **64.2**. `both invalid` n=53 (10.4%): 1m 99.2 / 680k 101.6. `1m valid & 680k invalid` n=47: keep 1m 52.5.

**Conclusions (please follow):**
1. **Do NOT run 680k on low-confidence rows.** Because the winning gate always prefers 1m when 1m is valid, 680k is only *used* on **1m-invalid** rows (~20% ≈ 95–100k test imgs). Running 680k on exactly those captures the full **-4.8** gain in **~40 min** at 4 procs. My `select_rescue.py` must be run with a **"invalid only"** mode (I'll add it; `--q 0` is NOT enough because it still keeps lowconf).
2. **Do NOT use score/formula/3-vote selection among both-valid** — all measured *worse* than plain 1m (35.6–35.8 vs 35.4). Keep it simple.
3. **Both-invalid rows (10.4%, 1m≈99) are the remaining lever**: use rigel (currently ~equal, will improve with training) and/or rotation-TTA. Rotation-TTA conf-select was worse globally but helped `rot0`-empty rows; test it **only on both-invalid** (after 680k) and take it only if it beats the fallback.
4. `extra`-dict snapping remains invalid; RDKit round-trip is safe (rigel's 99.99% test) [[rigel_mu_inchi_roundtrip.md]].

## Execution (I will run step 2 unless altair objects)
1. ~12:10 when 1m done: `select_rescue.py --invalid-only` -> `rescue_ids.csv` (~97k).
2. I launch `infer_chunk.py --ckpt 1m680k --ids rescue_ids.csv --nshard 4 --bs128 --conf` -> `preds680/`.
3. Assembly: `assemble_final.py --ms preds/shard*.csv --ms2 preds680/shard*.csv --rigel rigel_test.csv` with the **validity gate**.
4. Afternoon (13:00–15:30): rigel resumes training (33GB) + altair rotation-TTA on **both-invalid** rows; then rigel infers only on both-invalid.

Tags: #ensembling #insight #plan
