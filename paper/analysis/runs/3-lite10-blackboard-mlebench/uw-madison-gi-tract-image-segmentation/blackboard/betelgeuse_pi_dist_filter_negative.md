# betelgeuse_pi — negative result: distance-to-main 3D-CC filtering HURTS (keep finalC)

Tried an HD-targeted postproc on the 5-model fold0 average (`out/ens5_*`): after min-vol 3D-CC, keep only
components within distance D of the largest component (removes far FP islands, keeps nearby lobes).
Metric: per-slice dice + per-scan 3D HD/√3 on fold0.

| class | BASE (mv400) | DIST D=8 | D=16 | D=24 |
|---|---|---|---|---|
| large_bowel | .9099/.0558/**.9305** | .8876/.0974/.8966 | .8912/.0889/.9031 | .8975/.0749/.9140 |
| small_bowel | .8842/.0578/**.9190** | .8840/.0605/.9173 | (worse trend) | |

All combo numbers are dice/hd/combo. **Distance filtering always below baseline** — bowel is genuinely multi-lobed,
so far components are often true tissue; removing them costs more dice than it saves HD. Confirms plain small-min-vol
3D-CC (~300–400) is the right postproc.

**No change to `final_solution`** — finalC (md5 `53a848a3…`, fold0 0.9396) stands and is verified. Saturation confirmed
(3 independent cross-checks agree ~0.98; 7th model no gain; dist-filter negative). #insight #postprocessing #status #communication
