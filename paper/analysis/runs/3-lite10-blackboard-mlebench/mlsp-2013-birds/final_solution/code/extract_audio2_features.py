import numpy as np, librosa, soundfile as sf, re
ROOT = '/download_job_code/payload/sandbox'
ED = f'{ROOT}/data/essential_data'

def load_ids():
    order = []
    for line in open(f'{ED}/rec_labels_test_hidden.txt'):
        line = line.strip()
        if line.startswith('rec_id'):
            continue
        order.append(int(line.split(',')[0]))
    return order

def load_map():
    m = {}
    for line in open(f'{ED}/rec_id2filename.txt'):
        line = line.strip()
        if line.startswith('rec_id'):
            continue
        rid, fn = line.split(',')
        m[int(rid)] = fn
    return m

def st(a):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    out = [a.mean(1), a.std(1), np.percentile(a, 10, axis=1), np.percentile(a, 50, axis=1),
           np.percentile(a, 90, axis=1), a.max(1)]
    return np.concatenate(out).astype(np.float32)

def extract(path):
    x, sr = sf.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    x = x - x.mean()
    x = x / (np.max(np.abs(x)) + 1e-9)
    hop = 512; nfft = 1024
    S = np.abs(librosa.stft(x, n_fft=nfft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=nfft)
    feats = []

    # harmonic / percussive split
    H, P = librosa.decompose.hpss(S, margin=2.0)
    he = H.sum(0); pe = P.sum(0); tot = he + pe + 1e-9
    feats.append(st(he / tot))
    feats.append(st(np.log1p(H.sum(0))))
    feats.append(st(np.log1p(P.sum(0))))

    # frequency band energy fractions (over time)
    bands = [(50, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]
    totf = S.sum(0) + 1e-9
    for lo, hi in bands:
        idx = (freqs >= lo) & (freqs < hi)
        feats.append(st(S[idx].sum(0) / totf))

    # spectral flux (onset strength) on mel
    Sm = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=96, fmin=50, fmax=8000)
    logSm = librosa.power_to_db(Sm, ref=np.max)
    flux = np.maximum(0, np.diff(logSm, axis=1))
    feats.append(st(flux))
    onset_env = librosa.onset.onset_strength(S=logSm, sr=sr, hop_length=hop)
    feats.append(st(onset_env))
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
    feats.append(np.array([len(onsets), len(onsets) / 10.0, np.mean(np.diff(onsets)) if len(onsets) > 1 else 0.0]))

    # active-frame focused descriptors (bird events instead of background)
    rms = librosa.feature.rms(S=S, frame_length=nfft)[0]
    for q in (60, 80):
        thr = np.percentile(rms, q)
        act = rms > thr
        if act.sum() < 3:
            act = rms >= np.sort(rms)[-3]
        feats.append(logSm[:, act].mean(1))
        feats.append(logSm[:, act].std(1))
        mf = librosa.feature.mfcc(S=logSm, n_mfcc=20)
        feats.append(mf[:, act].mean(1))
        feats.append(mf[:, act].std(1))

    # temporal modulation of mel bands (bird syllables have 1-10 Hz modulation)
    env = logSm - logSm.mean(1, keepdims=True)
    fft_env = np.abs(np.fft.rfft(env, axis=1))
    fr = np.fft.rfftfreq(env.shape[1], d=hop / sr)
    for lo, hi in [(0.5, 2), (2, 5), (5, 10), (10, 30)]:
        m = (fr >= lo) & (fr < hi)
        feats.append(fft_env[:, m].mean(1))
        feats.append(fft_env[:, m].std(1))
    return np.concatenate(feats).astype(np.float32)

if __name__ == '__main__':
    order = load_ids(); fnmap = load_map()
    X = []
    for i, rid in enumerate(order):
        X.append(extract(f'{ED}/src_wavs/{fnmap[rid]}.wav'))
        if i % 50 == 0:
            print('done', i, flush=True)
    X = np.stack(X)
    np.savez_compressed(f'{ROOT}/workspaces/pollux_beta/features_audio2.npz',
                        X=X, ids=np.array(order))
    print('X', X.shape, 'nan', np.isnan(X).sum())
