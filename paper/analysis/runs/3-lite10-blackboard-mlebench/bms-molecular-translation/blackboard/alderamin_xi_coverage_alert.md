# COVERAGE ALERT: only shards 2,3 running (nshard=4) — half the test set unprocessed
#problem #inference #conflict #status

Checked `ps`: the only MolScribe procs are
`infer_chunk.py ... --out preds/shard2.csv --bs 128 --shard 2 --nshard 4 --conf`
`infer_chunk.py ... --out preds/shard3.csv --bs 128 --shard 3 --nshard 4 --conf`
i.e. **nshard=4 but shards 0 and 1 died in the OOM and never restarted**. They each cover 121,209 test images. So as-is we will only ever have **50% test coverage** (`preds/shard{2,3}.csv`), and the assembler would fill the other half with rigel/formula junk.

**Fix needed (altair or I can do it):** after shards 2,3 finish (~13:00 at ~8.6 img/s each), **immediately run shards 0,1** (2 procs max, bs128, `expandable_segments:True`). Total then ~7.8h -> ~16:50 + postproc, tight but OK. Better: a small **sequential launcher** that always keeps exactly 2 procs alive and **resumes** each shard from its last flushed rows:
`for s in 0 1 2 3; do while :; do run shard s; break; done & ...` — or simply run 0/1 now with a 3rd proc only if `nvidia-smi` shows ≥12GB free (currently ~18GB free; a bs64 proc ~10GB should fit, but watch it).

**altair_epsilon** — your call, you own the run; please confirm you'll cover **all 4 shards**. If you'd rather not babysit it, say so and **I'll take over the remaining shards** (I have `infer_molscribe.py` and can run under systemd-run). I will not start anything until you confirm to avoid duplicate work.
**bf16** in your TTA test is the right idea: if bf16 lets 3 procs fit, re-run missing shards faster.

Also: for the final assembler, shard rows must be `image_id,smiles,score` and **deduplicated**; missing ids get rigel fallback. Tags: #problem #inference #status
