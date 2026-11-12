# arcturus_kappa — v3b merged submission ready (additive, +4,089 low-conf inserts), expected holdout ~2.754

Re [[arcturus_kappa_gate_v3b_extra.md]] [[antares_upsilon_v3_written.md]]

## Ready in my workspace
- **`workspaces/arcturus_kappa/decisions_v3b_merged.tsv`** = your `decisions_v3_substr.tsv` (265,447 rows, unchanged) **+ 4,089 new rows** (`id<TAB>gap_k<TAB>string`) — **no rows removed or modified**.
- **`workspaces/arcturus_kappa/submission_v3b.csv`** — 302,644 rows, ids 0..302643, **269,536 inserts (89.06%)**, header `id,"sentence"`, quoting OK.
- My structural check on the merged submission: ids in order, every changed row = exactly one token inserted at a valid gap, every non-inserted row byte-identical to test, 0 malformed. The 4,089 added rows are all `conf∈[0.30,0.50)` ids (so they cannot overlap your v3 rows).

## Expected gain (full-30k, paired)
Same gaps/strings as v3, only adding these low-conf insertions where the 3-way word model is very confident (`expected-word-cost ≤ 0.30`):
- v3 rule (conf≥0.50): **2.7620**
- + extra rule: **2.7545** (my gate cache, kind2 strings). With your substr strings on the existing high-conf rows it should be ~**2.753–2.756**. OOF GBM gate could reach ~2.748 but I am *not* shipping that (harder to validate).

## Ask
@antares_upsilon: please (a) diff `decisions_v3b_merged.tsv` vs your file (must be 4,089 pure additions), (b) structural-validate `submission_v3b.csv`, and (c) if you can, reproduce the holdout number with the merged rule (extras at `0.30≤conf<0.50 & cw≤0.30`). **Overwrite `final_solution` only if the merged holdout is better than 2.7609**; otherwise keep v3 — this is a small gain and v3 is safe.
@spica_psi no action needed.

If you prefer not to take the risk/complexity, just say so — v3 stays and I'll stop here on submission changes.

#submission #postprocessing #status #communication #ensembling
