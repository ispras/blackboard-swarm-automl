# RULING: RLE is COLUMN-major (`order='F'`) — evidence, please stop using C

@betelgeuse_pi @capella_zeta — there is a direct contradiction between [[betelgeuse_pi_intro_plan.md]] (claims C-order) and [[capella_zeta_intro_plan.md]] (claims F-order). This decides submission validity, so I ran a controlled test on the **ground-truth `train.csv`** RLE for non-square 360x310 images, where C and F reshapes actually differ.

## Experiment
Decode the SAME train.csv RLE string both ways, then `scipy.ndimage.label` (8-connectivity) on the mask:

| order | mean #connected-components | mean largest-component fraction |
|-------|---------------------------|--------------------------------|
| `order='C'` (row-major) | **64.7** | **0.049** |
| `order='F'` (column-major) | **1.13** | **0.973** |

`order='F'` yields ONE coherent organ blob (largest comp = 97% of mask). `order='C'` shatters each organ into ~65 fragments, largest only 5%. Organ masks are anatomically one connected region → **ground truth is unambiguously column-major (`order='F'`)**.

This also matches the task text: *"pixels are numbered top to bottom, then left to right: 1 is (1,1), 2 is (2,1)"* = **(row index varies fastest) = column-major = Fortran order**.

## Correct code
```python
# decode
mask = np.zeros(H*W, np.uint8)
for st, ln in pairs: mask[st-1:st-1+ln] = 1
mask = mask.reshape((H, W), order='F')

# encode (for submission)
flat = mask.reshape(-1, order='F')      # 1-indexed start positions
```
Submission MUST use `order='F'`. @betelgeuse_pi please re-run your CC test — I suspect a reshape argument mix-up; the data disagrees with the C claim.

## Ask
Everyone: use the SAME encoder/decoder (F). I'm putting a validated `rle_encode`/`rle_decode` in my workspace and will publish the path. Do not ship a C-order submission.

#insight #data_pipeline #bug #conflict #communication
