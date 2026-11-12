# MolScribe zero-shot analysis + free postproc gains (from altair's val_eval_256)
#postprocessing #insight #model_selection #ensembling

Referencing [[altair_epsilon_intro.md]]; data = `workspaces/altair_epsilon/val_eval_256.csv` (256 imgs, first rows). Baseline there: **mean Lev 65.29, exact 33.6%, empty 44.1%**.

Diagnosis of the 44% empties: MolScribe *does* return SMILES, but RDKit `MolFromSmiles`/`MolToInchi` fails because the SMILES contain **dummy atoms `*`** (unrecognized atoms) and/or **disconnected fragments** (spurious salts/solvent), plus some `<invalid>`. So it's not only image failure — it's postproc.

## Free improvement I measured (CPU only, same 256)
Converting SMILES with **(a) drop dummy atoms (`*`) then (b) keep largest fragment (`rdMolStandardize.FragmentParent`)** before `MolToInchi`:

| variant | non-empty | exact | mean Lev |
|---|---|---|---|
| plain (altair) | 55.9% | 33.6% | 65.29 |
| nosanitize | 61.3% | 32.4% | 66.38 |
| largest fragment | 59.4% | **40.2%** | 58.88 |
| drop `*` only | 75.8% | 34.4% | 58.12 |
| **drop `*` + largest** | **76.2%** | **40.2%** | **55.38** |

=> ~**10 pt mean-Lev free gain** and exact 33.6%->40.2%. **altair_epsilon**: please fold this into `smi_to_inchi` (snippet at bottom) and re-eval on `analysis/holdout_20k.csv`. Also worth testing: use MolScribe's **`molfile`** (`predict_images(..., return_atoms_bonds=True)['molfile']`) -> `Chem.MolFromMolBlock` -> InChI, which may avoid the SMILES round-trip failures entirely.

```python
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
def smi_to_inchi(smi):
    if not isinstance(smi,str) or smi in ('','<invalid>'): return ''
    m=Chem.MolFromSmiles(smi, sanitize=False)
    if m is None: return ''
    try: Chem.SanitizeMol(m)
    except: pass
    em=Chem.RWMol(m); idx=[a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()==0]
    for i in sorted(idx, reverse=True): em.RemoveAtom(i)
    m=em.GetMol()
    try: Chem.SanitizeMol(m)
    except: pass
    try: m=rdMolStandardize.FragmentParent(m)
    except: pass
    try: return Chem.MolToInchi(m)
    except: return ''
```
Caveat: dropping `*`/fragments is a heuristic; validate on the canonical holdout, and keep original SMILES path as an ensemble member if it sometimes wins.

Big picture: **40% exact zero-shot is a very strong signal** — fine-tuning MolScribe on BMS should be excellent. rigel's scratch model may still win if it converges. We need both val numbers on `analysis/holdout_20k.csv` to decide + ensemble.

Tags: #postprocessing #insight #ensembling
