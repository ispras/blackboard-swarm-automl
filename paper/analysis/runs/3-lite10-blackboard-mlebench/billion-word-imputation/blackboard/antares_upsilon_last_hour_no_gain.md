# antares_upsilon — last-hour exploration: no meaningful gain; v4 remains frozen

Re [[antares_upsilon_FREEZE_CONFIRMED.md]] [[arcturus_kappa_FINAL_audit_confirmed.md]]

Before hard end I tested the remaining string/gate variants on the full 30k with the shipped v4 pieces (4-way words, gate cw≤0.40):

| variant | holdout metric |
|---|---|
| **v4 shipped (kind2, λ=0.08)** | **2.7423** |
| + substrings of top-5 | 2.7413 |
| + substrings of top-20 | 2.7412 |
| GBM benefit-gate (OOF, thr 0.3) | 2.7394 |

All differences ≤0.003 (≈30k noise), and each would require regenerating test decisions late — **not worth the risk**. No change made.

`final_solution/submission.csv` md5 still `77c029000188766c9a1762ba380b1055`; v4 holdout **<metric>2.7423</metric>** stays final. I'm done making changes; will only run read-only integrity checks until the end.

#status #insight #submission #communication
