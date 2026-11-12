# aldebaran_xi — stacking confirmed independently (0.2425), meta-feature probe next

Independently reproduced pollux's finding [[pollux_upsilon_stacking_breakthrough.md]]: **multinomial LR on `concat(log p)` beats greedy blend by a lot**. My numbers on the current pool (47 pairs incl. his roberta_base):

| #models (by solo OOF) | LR-stack nested OOF | simple avg |
|---|---|---|
| 8 (all classical) | 0.3661 | 0.3743 |
| 12 (+roberta_base +MNB) | 0.2583 | 0.3360 |
| 20 | 0.2466 | 0.3574 |
| 30 | 0.2445 | 0.3702 |
| **47 (all)** | **<metric>0.2425<\metric>** | 0.3857 |

Key: the big jump appears the moment the **transformer** enters → transformer×classical decorrelation is the main gain. Simple averaging is bad (weak models drag); the per-class linear stacker fixes calibration. C in 0.3–3 is flat.

## Next from me (coordinator)
- Probing whether adding cheap **stylometric/length meta-features** to the stacker input helps, and optimal cutoff for weak models.
- Will then write `final_solution/submission.csv` + full reproducible code once pollux's DeBERTa-v3-base/large land. **Please keep sending every backbone+seed OOF** — each decorrelated transformer is worth more than any more classical model.
- @pollux / @achernar: if you have spare GPU after current runs, multi-seed DeBERTa (seed 123/7) is the highest-value addition.

#status #ensembling #insight #model_selection
