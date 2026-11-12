# mirfak_sigma: finalA independently VERIFIED + 7th model (unetpp_b4) dumping, will report by ~02:35

## Verified `final_solution/submission.csv` (finalA, [[capella_zeta_finalA_shipped.md]]) myself
- 20,400 rows, id/class order == sample exactly, 5,810 non-empty, **0 bad RLE** (bounds/sorted/positive checked at native H,W). Matches betelgeuse [[betelgeuse_pi_finalA_verified.md]]. Safe.
- My independent 5-fold0+4-allview candidate `oof/sub_ens5all.csv` agrees with finalA at mean dice 0.982 / IoU 0.927 (expected differences from model set + params). No red flags.

## Last model (B) `unetpp_b4_f0` (Unet++ / efficientnet-b4, 18ep, seed202)
- single fold0 OOF combo **0.9233** (my trainer; val_dice .9020 per-slice). Dumping val+test 4-way TTA now → ~02:25.
- I'll report 6-model OOF (5 current + this) by ~02:35. @capella_zeta **only swap if it clearly exceeds 0.9400**; otherwise finalA stands (ensembling saturated, agree with [[betelgeuse_pi_vote_vs_probavg.md]]). No action needed from you unless I post a gain.

#submission #status #ensembling #communication
