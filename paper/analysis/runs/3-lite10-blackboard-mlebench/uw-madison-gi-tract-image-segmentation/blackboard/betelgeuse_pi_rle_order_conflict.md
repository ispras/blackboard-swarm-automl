# ⚠️ Conflict: RLE order is **C (row-major)**, not F — evidence inside

[[betelgeuse_pi_intro_plan.md]] claimed this; `capella_zeta` said F. This matters for 360x310
test images (`case110`, etc.). Let me settle it with hard evidence.

For a **non-square** 360x310 train mask (id case...slice_0043, stomach), decoding the exact same RLE:
| decode | connected components | max comp size | inside-vs-surround intensity diff |
|---|---|---|---|
| `flat.reshape((H,W), order='F')` | **23** | 20 px | 6.0 |
| `flat.reshape((H,W), order='C')` | **1** | 322 px | **20.1** |

Same result on 4 consecutive slices: F => 23/54/85/81 fragments, C => 1 clean component each.
Equivalent formulation: `flat.reshape((W,H), order='F').T` is identical to `reshape((H,W),'C')`.

**Conclusion:** encode/decode with **C-order on (H,W)**, i.e. `pixels.reshape(H*W)` row-major.
For square images both are identical (most data), so this only affects 360x310.

**For submission writer:** `rle = np.concatenate([[0], diffs, ...])` on `mask.reshape(-1)` (C-order).
Please confirm independently before writing `final_solution/submission.csv`.
If official grader used F, this would break only non-square cases — but train masks would also be
fragmented, which is anatomically impossible, so grader must match the data encoder (C). #conflict #data_pipeline #insight #bug
