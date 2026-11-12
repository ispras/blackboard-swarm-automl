"""Independent sanity check of final_solution/submission.csv.
- header + 302,644 rows, ids exactly 0..302643
- CSV parses with doubled-quote escaping
- each output sentence is either the masked test sentence verbatim, or the
  masked tokens with exactly ONE extra whitespace token inserted
- reports identity/insertion counts and which ids were changed
Usage: python verify_submission.py [submission.csv]
"""
import csv, sys, collections

TEST = "/download_job_code/payload/sandbox/workspaces/alkaid_pi/test_v2.txt"
SUB = sys.argv[1] if len(sys.argv) > 1 else "/download_job_code/payload/sandbox/final_solution/submission.csv"

test = {}
with open(TEST, newline="", encoding="utf-8") as f:
    r = csv.reader(f, quotechar='"', doublequote=True)
    hdr = next(r)
    for row in r:
        test[int(row[0])] = row[1]

problems = collections.Counter()
ids = []
n_ident = n_ins = 0
max_ins = 0
with open(SUB, newline="", encoding="utf-8") as f:
    r = csv.reader(f, quotechar='"', doublequote=True)
    h = next(r)
    if h != ["id", "sentence"]:
        problems["bad_header"] += 1
        print("HEADER:", h)
    for row in r:
        if len(row) != 2:
            problems["bad_ncol"] += 1
            continue
        i = int(row[0]); s = row[1]
        ids.append(i)
        if i not in test:
            problems["unknown_id"] += 1
            continue
        a = test[i].split()
        b = s.split()
        if b == a:
            n_ident += 1
            continue
        # try to locate one extra token
        ok = False
        if len(b) == len(a) + 1:
            for k in range(len(b)):
                if b[:k] + b[k + 1:] == a:
                    ok = True; break
        if ok:
            n_ins += 1
        else:
            problems["not_one_insertion"] += 1

ids_set = set(ids)
missing = sorted(set(test) - ids_set)
dups = len(ids) - len(ids_set)
print(f"rows={len(ids)} expected={len(test)} identity={n_ident} insertion={n_ins}")
print(f"problems={dict(problems)} missing={len(missing)} dups={dups}")
print(f"id_range=[{min(ids)},{max(ids)}] unique={len(ids_set)}")
if missing[:10]:
    print("missing sample", missing[:10])
