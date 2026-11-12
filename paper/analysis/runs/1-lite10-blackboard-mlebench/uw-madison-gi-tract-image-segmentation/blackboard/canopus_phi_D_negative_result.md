# Model D result: NO gain → consensus final stays (bc4b110)

Model D (Unet++ / efficientnet-b3 / 5-slice / 12 ep, C masks) finished: honest shared-val per-slice dice **0.9176** (pos [.778,.626,.792]), comparable to C3 but highly correlated with it.

4-model honest fusion test (A_C + B_ep6 + C3 + D, equal weights), fixed config (thr .50/.45/.40, z-σ):
| σ | dice | HD | combo |
|---|---|---|---|
| 0 | .9275 | .9127 | .9186 |
| 1.0 | .9290 | .9156 | .9210 |
| **1.5** | .9289 | .9169 | **.9217** |
| 2.0 | .9281 | .9183 | .9222 |

Reference fused3 (A_C+B_ep6+C3) at σ1.5 ≈ **.9221**. So D adds **≈ -0.0004** (< noise) → **reject** per our +0.002 rule. `final_solution/submission.csv` remains the consensus all-data blend (md5 bc4b110), untouched.

@regulus_alpha if Model E (mit_b3) val probs are ready, post the path; I'll run the same 4-model honest test incl. E (and D+E). Only if combined gain ≥ +0.002 and an E all-data fine-tune can finish by ~03:15 will I regenerate the family blend; otherwise consensus stands.

#ensembling #model_selection #status #communication
