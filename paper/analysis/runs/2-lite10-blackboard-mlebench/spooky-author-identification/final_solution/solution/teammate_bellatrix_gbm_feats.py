import numpy as np, pandas as pd, time, json, os, re, string
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import lightgbm as lgb
import xgboost as xgb

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = '/download_job_code/payload/sandbox/data'
tr = pd.read_csv(f'{DATA}/train.csv'); te = pd.read_csv(f'{DATA}/test.csv')
classes = ['EAP', 'HPL', 'MWS']
y = tr['author'].map({c: i for i, c in enumerate(classes)}).values
Xtxt = tr['text'].fillna('').values; Ttxt = te['text'].fillna('').values
fold_id = np.load(os.path.join(OUT, 'fold_id.npy'))
folds = [(np.where(fold_id != k)[0], np.where(fold_id == k)[0]) for k in range(5)]
N, M = len(tr), len(te)

STOP = sorted(ENGLISH_STOP_WORDS)
PUNCT = list('.,;:"\'?!()-')
COMMON = ['the','of','and','to','a','in','that','it','was','he','she','i','his','her','as','with','had','which','by','on','for','at','not','but','from','this','be','an','are','were','or','my','me','you','had','have','would','could','one','all','so','if','no','there','their','them','we','when','who','what','been','but','him']

def feats(texts):
    rows = []
    for t in texts:
        words = t.split()
        wl = [len(w) for w in words] or [0]
        low = t.lower()
        d = {
            'n_chars': len(t), 'n_words': len(words),
            'mean_wlen': np.mean(wl), 'max_wlen': max(wl), 'std_wlen': np.std(wl),
            'n_uniq': len(set(low.split())), 'uniq_ratio': len(set(low.split())) / max(len(words), 1),
            'n_upper': sum(c.isupper() for c in t), 'upper_ratio': sum(c.isupper() for c in t) / max(len(t), 1),
            'n_digits': sum(c.isdigit() for c in t), 'n_space': t.count(' '),
            'n_stop': sum(w.lower() in ENGLISH_STOP_WORDS for w in words),
        }
        d['stop_ratio'] = d['n_stop'] / max(len(words), 1)
        for p in PUNCT:
            d['p_' + ('quote' if p == '"' else 'apos' if p == "'" else p)] = t.count(p)
        d['n_comma_per_word'] = t.count(',') / max(len(words), 1)
        d['n_semicolon_per_word'] = t.count(';') / max(len(words), 1)
        for c in COMMON:
            d['w_' + c] = low.split().count(c)
        rows.append(d)
    return pd.DataFrame(rows).astype(np.float32)

print('handcrafted..', flush=True)
Ftr = feats(Xtxt); Fte = feats(Ttxt)
print(Ftr.shape, flush=True)

t0 = time.time()
oof_lgb = np.zeros((N, 3)); tst_lgb = np.zeros((M, 3))
oof_xgb = np.zeros((N, 3)); tst_xgb = np.zeros((M, 3))
for k, (itr, iva) in enumerate(folds):
    tw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, strip_accents='unicode', max_features=60000)
    Xw = tw.fit_transform(Xtxt[itr]); Vw = tw.transform(Xtxt[iva]); Tw = tw.transform(Ttxt)
    tc = TfidfVectorizer(ngram_range=(3, 5), min_df=3, sublinear_tf=True, strip_accents='unicode', analyzer='char_wb', max_features=120000)
    Xc = tc.fit_transform(Xtxt[itr]); Vc = tc.transform(Xtxt[iva]); Tc = tc.transform(Ttxt)
    sw = TruncatedSVD(n_components=100, random_state=42); sw.fit(Xw)
    sc = TruncatedSVD(n_components=100, random_state=42); sc.fit(Xc)
    Fw_tr, Fw_va, Fw_te = sw.transform(Xw), sw.transform(Vw), sw.transform(Tw)
    Fc_tr, Fc_va, Fc_te = sc.transform(Xc), sc.transform(Vc), sc.transform(Tc)
    A = np.hstack([Ftr.iloc[itr].values, Fw_tr, Fc_tr])
    B = np.hstack([Ftr.iloc[iva].values, Fw_va, Fc_va])
    C = np.hstack([Fte.values, Fw_te, Fc_te])
    sca = StandardScaler(); A = sca.fit_transform(A); B = sca.transform(B); C = sca.transform(C)
    clf = lgb.LGBMClassifier(n_estimators=900, learning_rate=0.03, num_leaves=31, subsample=0.8,
                             colsample_bytree=0.7, reg_lambda=1.0, objective='multiclass', num_class=3,
                             n_jobs=4, random_state=42, verbosity=-1)
    clf.fit(A, y[itr], eval_set=[(B, y[iva])], callbacks=[lgb.early_stopping(80, verbose=False)])
    oof_lgb[iva] = clf.predict_proba(B); tst_lgb += clf.predict_proba(C) / 5
    xc = xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8,
                           colsample_bytree=0.7, reg_lambda=1.0, objective='multi:softprob', num_class=3,
                           tree_method='hist', n_jobs=4, random_state=42, early_stopping_rounds=60, eval_metric='mlogloss')
    xc.fit(A, y[itr], eval_set=[(B, y[iva])], verbose=False)
    oof_xgb[iva] = xc.predict_proba(B); tst_xgb += xc.predict_proba(C) / 5
    print(f'fold {k} lgb {log_loss(y[iva], oof_lgb[iva]):.4f} xgb {log_loss(y[iva], oof_xgb[iva]):.4f} {time.time()-t0:.0f}s', flush=True)

res = {}
for name, o, t in [('lgbm_feats', oof_lgb, tst_lgb), ('xgb_feats', oof_xgb, tst_xgb)]:
    ll = log_loss(y, o, labels=[0, 1, 2]); res[name] = round(float(ll), 5)
    np.save(f'{OUT}/oof_{name}.npy', o); np.save(f'{OUT}/test_{name}.npy', t)
    print(name, ll, flush=True)
json.dump(res, open(f'{OUT}/gbm_results.json', 'w'), indent=2)
print(json.dumps(res, indent=2))
