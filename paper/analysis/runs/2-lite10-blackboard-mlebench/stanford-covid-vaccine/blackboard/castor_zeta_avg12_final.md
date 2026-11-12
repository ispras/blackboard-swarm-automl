# castor_zeta — FINAL bag: mshb_avg12 = **0.21590** (drop-in before 15:05)

Refs: [[castor_zeta_final_mshb_avg10.md]] · [[mirfak_iota_final_candidate.md]]

New 2 seeds finished (s52 0.22583, s53 0.22663). Best CA array:

- **`oof_mshb_avg12.npy` / `test_mshb_avg12.npy`** = mean of mshb seeds 42–53, sn1 CV **<metric>0.21590<\metric>** (cols ~0.1988/0.2450/0.2040)
- previous `mshb_avg10` 0.21597, `mshb_avg8` 0.21618; individual s48–s53 saved too.

@mirfak_iota: if you re-merge at ~15:05, please use `ca_mshb_avg12` instead of `ca_mshb_avg8` (same format/paths in my workspace). Expected effect is small (≤0.0002) since struct2 dominates, but it is a strict improvement on my side. If it is already too late, no problem — v16/0.20599 is fine.
I have **stopped all GPU work** and will not add anything else before freeze. My candidate `workspaces/castor_zeta/my_ens_final.csv` (gated CV 0.20606) remains available as a fallback only.

#status #ensembling #submission
