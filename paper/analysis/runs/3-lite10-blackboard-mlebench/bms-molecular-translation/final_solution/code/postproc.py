import re
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
_VALID=set()
for _z in range(1,119): _VALID.add(Chem.GetPeriodicTable().GetElementSymbol(_z))
_VALID|={'D','T'}
_BR=re.compile(r'\[([^\]]*)\]')
def _elem(inner):
    m=re.match(r'\d*([A-Z][a-z]?|[a-z])', inner); return m.group(1) if m else ''
def sanitize(smi):
    if not isinstance(smi,str) or not smi: return ''
    smi=smi.replace('*','C')
    smi=_BR.sub(lambda m: m.group(0) if _elem(m.group(1)) in _VALID else 'C', smi)
    smi=re.sub(r'\[,\]','C',smi)
    return smi
def largest(smi):
    parts=[p for p in smi.split('.') if p and p not in ('[.]','*','[H]')]
    best='';bn=-1
    for p in parts:
        try:
            m=Chem.MolFromSmiles(p,sanitize=False)
            if m is None: continue
            n=m.GetNumAtoms()
        except Exception: continue
        if n>bn: bn=n;best=p
    return best
def _mol(smi):
    if not isinstance(smi,str) or not smi or smi=='<invalid>': return None
    smi=largest(sanitize(smi))
    if not smi: return None
    try:
        m=Chem.MolFromSmiles(smi,sanitize=False)
        if m is None: return None
        try: Chem.SanitizeMol(m)
        except: pass
        return m
    except Exception:
        return None
def smi_to_inchi(smi):
    m=_mol(smi)
    if m is None: return ''
    try: return Chem.MolToInchi(m)
    except Exception: return ''
def smi_to_inchi_fallback(smi):
    m=_mol(smi)
    if m is None: return ''
    try:
        inchi=Chem.MolToInchi(m)
        if inchi: return inchi
    except Exception: pass
    try:
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        f=CalcMolFormula(m)
        return 'InChI=1S/'+f if f else ''
    except Exception: return ''
def lev(a,b):
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(prev[j]+1,cur[j-1]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]
