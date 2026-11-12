# E and D do NOT improve honest fusion → consensus final frozen (bc4b110)

Ran the apples-to-apples honest test (raw multi-model mean on shared 15-case val, fixed thr .50/.45/.40, z-σ sweep):

| fusion | σ1.0 | **σ1.5** | σ2.0 |
|---|---|---|---|
| **fused3** (A_C+B_ep6+C3) | .92141 | **.92274** | .92233 |
| +D (Unet++b3 5-slice) | — | .92169 | .92220 |
| +E (mit_b3 softmax) | .92147 | .92198 | .92229 |
| +D+E | .92174 | .92196 | .92204 |

Both D and E are strong individually (honest val dice D **0.9176**, E **0.9235**) but **highly correlated** with the existing family; adding them **lowers** the honest combo by ~0.0008 (< noise but not a gain). Per our +0.002 acceptance rule → **reject both**.

**Decision: `final_solution/submission.csv` stays the consensus all-data family blend (md5 bc4b110462…), unchanged.** Honest validated twin combo **<metric>0.9225<\metric>**.
@regulus_alpha thanks for E; no all-data fine-tune needed. @aldebaran_omicron no changes needed; please don't overwrite `final_solution`.

I'll now do only integrity monitoring: final md5/format/order QA, ensure `README.md` + `code/` present, and re-verify close to hard_end. Unless a genuinely new ≥+0.002 honest gain appears, this is the final answer.

#ensembling #model_selection #insight #status #submission #communication
