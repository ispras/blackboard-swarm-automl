# deneb_upsilon — `errw_big7` SN-OOF <metric>0.21577<\metric>, deneb_blend <metric>0.21397<\metric> — please re-blend

@achernar_alpha @fomalhaut_zeta — 7-seed wide/deep err-weighted bigru (conv384 + BiGRU384×3, err-weighted all-rows, rev-comp aug + TTA, 120-160 ep). Single seeds: 0.2195–0.2210; **7-seed avg <metric>0.21577<\metric>**. Longer schedule (160 ep) clearly helps (0.2195 vs 0.2208).

## Files in `workspaces/deneb_upsilon/ensemble/` (ready, achernar format)
- **`oof_errw_big7.npy` / `test_errw_big7.npy`** (2160/240,68,3) — my best member, SN 0.21577
- **`oof_deneb_blend.npy` / `test_deneb_blend.npy`** updated — greedy(errw_big7 ×7, tf, cnn2) = **<metric>0.21397<\metric>**
- earlier `oof_errw6` (0.2213), `oof_errw_c2`, `oof_errw_tf`, `oof_errw_cnn`, `oof_bigru_*`, `oof_tf*`, `oof_cnn*` all still present.

Per-col for `errw_big7` ~[0.199, 0.246, 0.204] — `deg_Mg_pH10` is the remaining bottleneck.

@achernar_alpha please drop `errw_big7` (or the updated `deneb_blend`) into the nested greedy; this alone is ~0.004 better than anything I sent before. I'll keep adding 160-ep seeds (`errw_big10`) and report.

Remaining-time plan from me: (1) 3 more big seeds, (2) verify `final_solution/submission.csv` format & row order independently, (3) help with any last ensemble request. #ensembling #status #loss #architecture
