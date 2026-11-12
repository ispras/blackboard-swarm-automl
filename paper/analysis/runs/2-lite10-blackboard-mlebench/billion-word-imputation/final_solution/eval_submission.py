"""Shared validation scorer for Billion Word Imputation (betelgeuse_tau).

HOLDOUT SPEC (for the whole swarm):
  Source : last 20,000 non-empty lines of data/train_v2.txt (streamed from the zip),
           filtered to >=4 tokens -> 19,948 sentences.
  Files  : workspaces/betelgeuse_tau/val_truth.jsonl  (id, gap, word, orig)
           workspaces/betelgeuse_tau/val_test.csv     (id, sentence with word deleted)
  Deletion: random interior token (never first, never last). Seed 1234.
  IMPORTANT: do NOT train on these last 20k lines (use lines before them).

Scorer:
  python eval_submission.py --pred submission.csv
    submission.csv must have columns id,sentence (header id,"sentence" like the task).
  Prints mean Levenshtein, % exact, and mean cost split by whether we changed the sentence.

Candidate-applier (evaluate a gap,word candidate file):
  python apply_candidates.py --cand cands.tsv --out sub.csv
    cands.tsv columns: id<TAB>gap<TAB>word   (gap = insertion index into the DELETED
    sentence tokens, 0..n meaning insert before token[gap]; use empty word = no insert).
"""
import argparse, csv, json, sys, statistics


def load_truth(path='val_truth.jsonl'):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def insert_at(sentence, gap, word):
    toks = [t for t in sentence.split(' ') if t != '']
    if word == '' or word is None:
        return ' '.join(toks)
    gap = max(0, min(gap, len(toks)))
    return ' '.join(toks[:gap] + [word] + toks[gap:])


def score(pred_rows, truth):
    import Levenshtein
    by_id = {int(r['id']): r for r in pred_rows}
    ds, changed, unchanged = [], [], []
    for d in truth:
        i = d['id']
        if i not in by_id:
            raise SystemExit(f'missing id {i}')
        pred = by_id[i]['sentence']
        dist = Levenshtein.distance(d['orig'], pred)
        ds.append(dist)
        # compare token-set to val deleted sentence
        if pred.strip().split(' ') == d['orig'].strip().split(' '):
            pass
        (changed if pred != by_id[i].get('_test', pred) else unchanged).append(dist)
    return statistics.mean(ds), ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--truth', default='val_truth.jsonl')
    ap.add_argument('--test', default='val_test.csv')
    ap.add_argument('--show-errors', type=int, default=8)
    args = ap.parse_args()

    truth = load_truth(args.truth)
    test_rows = list(csv.DictReader(open(args.test, encoding='utf-8')))
    test_by_id = {int(r['id']): r['sentence'] for r in test_rows}
    pred_rows = list(csv.DictReader(open(args.pred, encoding='utf-8')))
    import Levenshtein
    by_id = {int(r['id']): r['sentence'] for r in pred_rows}

    ds, d_changed, d_unchanged = [], [], []
    exact = 0
    gap_hits = 0
    examples = []
    for d in truth:
        i = d['id']
        pred = by_id[i]
        dist = Levenshtein.distance(d['orig'], pred)
        ds.append(dist)
        if dist == 0:
            exact += 1
        base = test_by_id[i]
        if pred == base:
            d_unchanged.append(dist)
        else:
            d_changed.append(dist)
            # infer inserted tokens / gap for diagnostics
            examples.append((dist, base, pred, d['word'], d['gap']))
    print(f'mean_levenshtein = {statistics.mean(ds):.4f}   (n={len(ds)})')
    print(f'exact_match      = {exact / len(ds):.4%}')
    print(f'changed rows     = {len(d_changed)} ({len(d_changed)/len(ds):.2%})')
    if d_changed:
        print(f'  mean dist | changed   = {statistics.mean(d_changed):.3f}')
    if d_unchanged:
        print(f'  mean dist | unchanged = {statistics.mean(d_unchanged):.3f}')
    examples.sort()
    if args.show_errors:
        print('\nlowest-distance changed examples:')
        for dist, base, pred, w, g in examples[:args.show_errors]:
            print(f'  d={dist} true={w!r} gap={g} | {pred[:120]}')


if __name__ == '__main__':
    main()
