import torch, numpy as np
from transformers import AutoTokenizer

def build_candidate_vocab(tok):
    vocab=tok.get_vocab()
    ids=[]; words=[]
    special=set(tok.all_special_ids)
    for t,i in vocab.items():
        if i in special: continue
        if not t.startswith('Ġ'): continue
        s=tok.convert_tokens_to_string([t]).strip()
        if s=='' or ' ' in s: continue
        ids.append(i); words.append(s)
    # punctuation may also appear non-G-space in mid-sentence? add common punctuation forms
    extra=[',','.',';',':','!','?','"','\'s','\'','--','-','(',')','%','$','&','@','#','``',"''"]
    for s in extra:
        for form in (s,):
            ii=tok.convert_tokens_to_ids('Ġ'+s)
            if ii is not None and ii!=tok.unk_token_id and ii not in ids:
                ids.append(ii); words.append(s)
    return np.array(ids), words

def insert_gap_seq(tokens, g, mask):
    return ' '.join(tokens[:g]) + ' ' + mask + ' ' + ' '.join(tokens[g:])
