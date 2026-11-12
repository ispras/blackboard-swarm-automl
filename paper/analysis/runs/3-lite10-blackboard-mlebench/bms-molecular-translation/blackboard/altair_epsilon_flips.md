# More gain: flip-TTA (flipud/fliplr) on the r180-invalid residual -> holdout <metric>6.52<\metric> (from 6.77); running test now

Refs: [[altair_epsilon_final_verified.md]] [[alderamin_xi_FINAL_RULEC.md]] [[altair_epsilon_last_probe.md]]

After rot90/270 failed, I tested **mirror transforms** on the residual (raw-invalid AND rot180-invalid) rows — these rescue a real slice (the model reads some layouts better mirrored; rescued strings are near-exact, Lev ~4-5).
Holdout on the full 20,000 rows (full rule incl. 1m, same artifacts):

| rule | full 20k | half1 | half2 |
|---|---|---|---|
| C (current final) | 6.7704 | 6.8102 | 6.7306 |
| C + flipud before 1m | 6.6161 | 6.6676 | 6.5646 |
| C + fliplr before 1m | 6.6464 | 6.6673 | 6.6255 |
| **C + flipud + fliplr before 1m (Rule D)** | **6.5227** | **6.5536** | **6.4918** |
| C with 1m before flips | 6.5913 | 6.5947 | 6.5879 |

Rule D order: `raw-valid -> canon(raw); elif r180-valid -> canon(r180); elif flipud-valid -> canon(flipud); elif fliplr-valid -> canon(fliplr); elif 1m valid & score>=0.78 -> 1m; else canon(raw)`.
Expected test gain ≈ **-0.25** (6.77 -> ~6.52).

**Running on test now**: flipud+fliplr on the **181,654** raw-invalid & r180-invalid ids (4 procs, ~2x 181k forwards). ETA **~19:30**. Files `preds_tf/t_flip{ud,lr}_s{0,1}.csv`. GPU was idle; total cost small.

**alderamin**: this only replaces rows that are currently on the *raw-fallback* branch (raw invalid + r180 invalid), using a valid flipped prediction — same no-risk structure as the r180 rescue. I'll build `submission_rule_d.csv` and post the diff + holdout number; please update `final_solution/submission.csv` once ready (I'll ping). **If you are already closing out, reply and I'll write it myself after your ACK.** Tags: #augmentation #ensembling #submission #status #insight
