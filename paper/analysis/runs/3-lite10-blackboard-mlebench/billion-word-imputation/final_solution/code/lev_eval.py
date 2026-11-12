"""Common holdout eval harness for Billion Word Imputation.
Predictions: iterable/sequence of full sentence strings, aligned with
holdout_gap.txt order. Score = mean char-level Levenshtein vs holdout_orig.txt.
Usage:
    from lev_eval import score_files, score_preds, load_holdout
    print(score_files('my_preds.txt'))
"""
import Levenshtein

BASE = '/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout'

def load_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [l.rstrip('\n') for l in f]

def load_holdout():
    return load_lines(f'{BASE}/holdout_orig.txt'), load_lines(f'{BASE}/holdout_gap.txt')

def score_preds(preds, orig=None):
    if orig is None:
        orig = load_lines(f'{BASE}/holdout_orig.txt')
    assert len(preds) == len(orig), (len(preds), len(orig))
    tot = 0
    for p, o in zip(preds, orig):
        tot += Levenshtein.distance(p, o)
    return tot / len(orig)

def score_files(pred_path):
    return score_preds(load_lines(pred_path))

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        print('score', score_files(sys.argv[1]))
    else:
        orig, gap = load_holdout()
        print('baseline (no insertion):', score_preds(gap, orig))
