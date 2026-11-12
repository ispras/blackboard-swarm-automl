"""Perch v2 on spectral-gated (denoised) audio, 5s@32kHz, 2.5s hop. Writes target probs."""
import os, time
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np, pandas as pd, librosa, tensorflow as tf

base = '/download_job_code/payload/sandbox'
WS = f'{base}/workspaces/arcturus_kappa'
MODEL = f'{base}/workspaces/castor_rho/models/perch2/Perch2.tflite'
LABELS = f'{base}/workspaces/castor_rho/models/perch2/perch2_class_labels.txt'
rec2f = {}
for line in open(f'{base}/data/essential_data/rec_id2filename.txt').read().splitlines()[1:]:
    a, b = line.split(','); rec2f[int(a)] = b
labels = pd.read_csv(LABELS, header=None).iloc[:, 0].tolist(); idx = {s: i for i, s in enumerate(labels)}
SCI = ["Certhia americana","Troglodytes pacificus","Empidonax difficilis","Sitta canadensis","Junco hyemalis",
       "Contopus cooperi","Catharus guttatus","Poecile rufescens","Ixoreus naevius","Setophaga occidentalis",
       "Catharus ustulatus","Empidonax hammondii","Piranga ludoviciana","Pheucticus melanocephalus",
       "Regulus satrapa","Vireo gilvus","Geothlypis tolmiei","Cyanocitta stelleri","Chordeiles minor"]
tgt = [idx[s] for s in SCI]

it = tf.lite.Interpreter(model_path=MODEL, num_threads=4); it.allocate_tensors()
BATCH = 8; it.resize_tensor_input(0, [BATCH, 160000]); it.allocate_tensors()
SR = 32000; W = 5 * SR; H = int(2.5 * SR)

def denoise(y16):
    D = librosa.stft(y16, n_fft=1024, hop_length=256)
    mag, phase = np.abs(D), np.angle(D)
    noise = np.percentile(mag, 15, axis=1, keepdims=True)
    mask = np.clip((mag - 1.5 * noise) / (mag + 1e-8), 0.0, 1.0)
    # light temporal smoothing of the mask
    mask = np.apply_along_axis(lambda r: np.convolve(r, np.ones(5) / 5, mode='same'), 1, mask)
    y = librosa.istft(mag * mask * np.exp(1j * phase), hop_length=256, length=len(y16))
    return y.astype(np.float32)

ids = sorted(rec2f); clips = []; owner = []
for rid in ids:
    y16, _ = librosa.load(f'{base}/data/essential_data/src_wavs/{rec2f[rid]}.wav', sr=16000, mono=True)
    y = librosa.resample(denoise(y16), orig_sr=16000, target_sr=SR)
    if len(y) < W: y = np.pad(y, (0, W - len(y)))
    starts = list(range(0, len(y) - W + 1, H))
    if starts[-1] + W < len(y): starts.append(len(y) - W)
    for st in starts: clips.append(y[st:st + W].astype('float32')); owner.append(rid)
print('clips', len(clips), flush=True)
P = {r: [] for r in ids}; t0 = time.time()
for s in range(0, len(clips), BATCH):
    b = clips[s:s + BATCH]; n = len(b)
    while len(b) < BATCH: b.append(np.zeros(W, dtype='float32'))
    it.set_tensor(0, np.stack(b)); it.invoke()
    lab = None
    for d in it.get_output_details():
        t = it.get_tensor(d['index'])
        if t.ndim == 2 and t.shape[-1] == 14795: lab = t
    for j in range(n):
        P[owner[s + j]].append(1 / (1 + np.exp(-lab[j, tgt])))
    if s % 80 == 0: print(s, round(time.time() - t0, 1), flush=True)
rows = []
for rid in ids:
    A = np.stack(P[rid])
    d = {'rec_id': rid}
    for k in range(19):
        d[f'pred_{k}'] = float(A[:, k].max())
    rows.append(d)
pd.DataFrame(rows).sort_values('rec_id').to_csv(f'{WS}/perch_dn_probs.csv', index=False)
print('DONE', round(time.time() - t0, 1), flush=True)
