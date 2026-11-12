import csv, random, statistics, collections
from rapidfuzz.distance import Levenshtein

rows=[]
with open('holdout_20k.tsv',encoding='utf-8') as f:
    r=csv.DictReader(f,delimiter='\t')
    for d in r: rows.append(d)
print("rows",len(rows))
rng=random.Random(7)

# common words from the holdout deleted tokens for a "guess common word" sim
common=['the','a','of','to','and','in','is','that','for','it','was','on','as','with','he','she','said','at','by','from','his','her','be','has','had','but','not','are','have','this','will','an','they','you','we','or','which','their','its','one','all','would','there','can','more','new','up','out','about','been','who','when','also','than','into','after','what','some','could','other','over','just','two','first','may','these','like','now','years','time','no','only','my']

def cost(sub,orig): return Levenshtein.distance(sub,orig)

id_costs=[]; wronggap_costs=[]; rightwrongword_costs=[]; oracle=0
rightword_short=[]
for d in rows:
    toks=d['masked'].split(); true=d['deleted']; p=int(d['pos']); orig=d['original']
    n=len(toks)+1
    id_costs.append(cost(d['masked'],orig))
    # true word at true gap
    sub=' '.join(toks[:p]+[true]+toks[p:])
    assert cost(sub,orig)==0, (sub,orig)
    # true word at a random wrong gap
    cands=[g for g in range(1,n-1) if g!=p]
    if cands:
        g=rng.choice(cands)
        sub=' '.join(toks[:g]+[true]+toks[g:])
        wronggap_costs.append(cost(sub,orig))
    # a random common word at true gap
    w=rng.choice(common)
    sub=' '.join(toks[:p]+[w]+toks[p:])
    rightwrongword_costs.append(cost(sub,orig))
    # weighted average over common words at true gap (expectation)
    rightword_short.append(statistics.mean(cost(' '.join(toks[:p]+[w]+toks[p:]),orig) for w in common[:30]))

print("identity mean",statistics.mean(id_costs),"median",statistics.median(id_costs))
print("true word at WRONG gap mean",statistics.mean(wronggap_costs),"median",statistics.median(wronggap_costs))
print("random common word at RIGHT gap mean",statistics.mean(rightwrongword_costs),"median",statistics.median(rightwrongword_costs))
print("avg-over-common at right gap mean",statistics.mean(rightword_short))
# deleted word length
lens=[len(d['deleted']) for d in rows]
print("deleted len mean",statistics.mean(lens),"median",statistics.median(lens))
# word-level identity distance
print("word-level identity mean distance", 1.0)
# if oracle gap but guess common word at right gap: cost = LEV(common,true)
for w in ['the','a','of','to','and','']:
    cs=[cost(w,true) for true in [d['deleted'] for d in rows]]
    cs=[cost(' '.join(d['masked'].split()[:int(d['pos'])]+([w] if w else [])+d['masked'].split()[int(d['pos']):]),d['original']) for d in rows]
    print(f"fixed word '{w}' at RIGHT gap: mean {statistics.mean(cs):.3f}")
# uniform random wrong-gap identity? no
