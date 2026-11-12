"""RLE convention helpers.

Facts (verified on data/train.csv):
- GT RLE is row-major: true mask M = rle_flat.reshape(H, W, order='C').
- Submission must be encoded with the SAME convention as the model's TRAIN-decode
  convention D:  encoded_flat = pred_D.flatten(order=D).  Grader (C) then recovers M.
  * C-trained model -> encode order='C'
  * F-trained model -> encode order='F'
  (Never "fix" an F-trained submission by re-encoding it as C.)
"""
import numpy as np, cv2

def rle_encode(mask, order="C"):
    px = mask.flatten(order=order)
    px = np.concatenate([[0], px, [0]])
    runs = np.where(px[1:] != px[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(int(x)) for x in runs)

def rle_decode(rle, h, w, order="C"):
    if not isinstance(rle, str) or rle.strip() == "":
        return np.zeros((h, w), np.uint8)
    s = rle.split(); st = np.asarray(s[0::2], np.int64) - 1; ln = np.asarray(s[1::2], np.int64)
    flat = np.zeros(h * w, np.uint8)
    for a, b in zip(st, ln):
        flat[a:a + b] = 1
    return flat.reshape(h, w, order=order)

def prob_to_true_320(prob320, orig_h, orig_w, train_conv="C"):
    """Map a (320,320) probability map from a model trained with convention
    `train_conv` into the TRUE-mask space, returned on the 320 grid."""
    p = cv2.resize(prob320.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    if train_conv.upper() == "F":
        p = p.flatten(order="F").reshape(orig_h, orig_w, order="C")
    return cv2.resize(p, (320, 320), interpolation=cv2.INTER_LINEAR)

def mask_true_to_submission(mask320, orig_h, orig_w):
    """C-trained / true-space mask -> resize to orig -> RLE encoded order='C'."""
    m = cv2.resize(mask320.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return rle_encode(m, order="C")
