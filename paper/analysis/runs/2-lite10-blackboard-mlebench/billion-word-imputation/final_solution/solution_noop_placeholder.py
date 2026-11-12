"""PLACEHOLDER safety submission: submit the test sentences unchanged (no insertion).

Guarantees a valid, parseable submission. Will be replaced by the agreed swarm
solution (gap localization + word imputation + decision-theoretic insertion).
"""
import csv, io, zipfile

SRC = 'data/test_v2.txt.zip'
DST = 'final_solution/submission.csv'

with zipfile.ZipFile(SRC) as z, z.open('test_v2.txt') as raw, \
        open(DST, 'w', encoding='utf-8', newline='') as g:
    rd = csv.reader(io.TextIOWrapper(raw, encoding='utf-8'))
    wr = csv.writer(g, quoting=csv.QUOTE_ALL)
    wr.writerow(['id', 'sentence'])
    next(rd)
    for row in rd:
        wr.writerow([row[0], row[1]])
print('wrote', DST)
