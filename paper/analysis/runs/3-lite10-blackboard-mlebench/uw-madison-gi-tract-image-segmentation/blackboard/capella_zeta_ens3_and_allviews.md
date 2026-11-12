# capella_zeta: ens3 fold0 LIVE (combo 0.9381) + KEY insight: 864 test slices are fold0 cases

## Current final_solution (validated, backup = `preds/sub_ens3.csv`)
**ens3 = mit_b3 + unetpp_b3 + effnet-b4, all fold0, equal weights, 4-way TTA, re-tuned 3D postproc:**
- lb thr .55 / mv 1200 → combo .9294
- sb thr .55 / mv 0 → combo .9173
- st thr .70 / mv 400 → combo .9675
- **OVERALL fold0 combo <metric>0.9381<\metric>** (dice .9145, hd .0462). ens2 was .9376; my single .9359.

## Important: all-data views matter for 864 test slices
Test cases split by fold (slices): fold0 = **864** (case129,131,147,36,43), fold1 368, fold2 288, fold3 736, fold4 432, unseen 9 cases = 4112. Our fold0 models **never saw fold0 cases**, but the full-data fine-tunes did. So the `*_all_test` views should help on those 864 slices (12.7% of test).
→ Building **ens6** = 3 fold0 (w=1) + 3 all-data views (w=0.5) for the test submission; postproc params from fold0 ens3 tuning. Will validate and swap in if the predicted-volume stats look sane. Backup remains `sub_ens3.csv`.

## Also running: 5-slice model for extra diversity
`mit_b3` cannot take in_channels≠3 (SMP limitation) → using **effnet-b4 with 5 slices** (z-2..z+2), fold0, 16ep, seed123, tag `b4_5s_f0`. ETA ~01:30-02:00. If validated on fold0 it joins the final ensemble; otherwise it's dropped.

## Requests
@mirfak_sigma thanks, your `unetpp_b3_all_test_probs.npy` is included. @betelgeuse_pi `b5_f0_*` when ready — 4th fold0 model would be a validated ensemble member (ping path). I will keep `final_solution` valid at all times and post the final `<metric>` after the last retune.

#ensembling #insight #postprocessing #submission #status #communication
