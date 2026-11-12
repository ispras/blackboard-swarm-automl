import os, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

ROOT = '/download_job_code/payload/sandbox'
SMILES_CHARS = ['#','%','(',')','+','-','.','/','0','1','2','3','4','5','6','7','8','9','=',
                '@','B','C','F','H','I','N','O','P','S','[','\\',']','b','c','i','l','n','o','p','r','s']
PAD, SOS, EOS = 0, 1, 2
ITOS = ['<pad>','<sos>','<eos>'] + SMILES_CHARS
STOI = {c:i for i,c in enumerate(ITOS)}
VOCAB = len(ITOS)
MAXLEN = 160
FALLBACK_INCHI = 'InChI=1S/C14H20N2O2/c1-11-3-5-12(6-4-11)14(17)16-7-8-18-10-13(16)9-15-2/h3-6,13,15H,7-10H2,1-2H3'

def encode_text(s, maxlen=MAXLEN):
    ids = [SOS] + [STOI[c] for c in s[:maxlen-1]] + [EOS]
    return ids

class InChIDataset(Dataset):
    def __init__(self, df, img_dir, size=256, train=False):
        self.ids = df.image_id.values
        self.texts = df.smiles.values
        self.img_dir = img_dir
        self.size = size
        self.train = train
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        iid = self.ids[i]
        p = os.path.join(self.img_dir, iid[0], iid[1], iid[2], iid+'.png')
        im = Image.open(p).convert('L')
        w, h = im.size
        m = max(w, h)
        if (w, h) != (m, m):
            n = Image.new('L', (m, m), 255)
            n.paste(im, ((m-w)//2, (m-h)//2))
            im = n
        im = im.resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.uint8))
        x = x.unsqueeze(0).repeat(3,1,1)
        t = torch.tensor(encode_text(self.texts[i]), dtype=torch.long)
        return x, t

def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ts = [b[1] for b in batch]
    L = max(len(t) for t in ts)
    out = torch.full((len(ts), L), PAD, dtype=torch.long)
    for i,t in enumerate(ts): out[i,:len(t)] = t
    return xs, out

def pos_enc_2d(h, w, d, device, dtype=torch.float32):
    assert d % 4 == 0
    d4 = d//4
    omega = torch.arange(d4, device=device, dtype=torch.float32) / d4
    omega = 1.0 / (10000 ** omega)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    out = torch.zeros(h*w, d, device=device, dtype=torch.float32)
    out[:, :d4] = torch.sin(ys[:,None]*omega[None,:]).repeat_interleave(w,0)
    out[:, d4:2*d4] = torch.cos(ys[:,None]*omega[None,:]).repeat_interleave(w,0)
    out[:, 2*d4:3*d4] = torch.sin(xs[:,None]*omega[None,:]).repeat(h,1)
    out[:, 3*d4:] = torch.cos(xs[:,None]*omega[None,:]).repeat(h,1)
    return out.to(dtype)

class Encoder(nn.Module):
    def __init__(self, d_model=512, pretrained=True, dropout=0.1):
        super().__init__()
        import timm
        m = timm.create_model('resnet34', pretrained=pretrained, in_chans=3)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.act1 if hasattr(m,'act1') else m.relu, m.maxpool,
                                  m.layer1, m.layer2, m.layer3)
        self.proj = nn.Conv2d(256, d_model, 1)
        self.d_model = d_model
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        f = self.stem(x)
        f = self.proj(f)
        B, d, H, W = f.shape
        pe = pos_enc_2d(H, W, d, f.device, f.dtype)
        f = f.flatten(2).transpose(1,2) + pe.unsqueeze(0)
        return self.drop(f)

class DecoderLayer(nn.Module):
    def __init__(self, d, nh, ffn, dropout=0.1):
        super().__init__()
        self.nh = nh; self.d = d
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.o1 = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.q2 = nn.Linear(d, d)
        self.kv2 = nn.Linear(d, 2*d)
        self.o2 = nn.Linear(d, d)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)
    def _split(self, x, n):
        return x.view(*x.shape[:-1], n, self.d // self.nh).transpose(-3, -2)
    def forward(self, x, mem, sa_cache=None, ca_cache=None, causal=False):
        B, L, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = self._split(q,self.nh), self._split(k,self.nh), self._split(v,self.nh)
        if sa_cache is not None:
            pk, pv = sa_cache
            k = torch.cat([pk, k], dim=-2); v = torch.cat([pv, v], dim=-2)
        new_sa = (k, v)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(causal and sa_cache is None))
        a = a.transpose(-3,-2).reshape(B, L, d)
        x = x + self.drop(self.o1(a))
        h = self.ln2(x)
        q2 = self._split(self.q2(h), self.nh)
        if ca_cache is not None:
            k2, v2 = ca_cache
        else:
            kv = self.kv2(mem); k2, v2 = kv.chunk(2, dim=-1)
            k2 = self._split(k2,self.nh); v2 = self._split(v2,self.nh)
        a2 = F.scaled_dot_product_attention(q2, k2, v2)
        a2 = a2.transpose(-3,-2).reshape(B, L, d)
        x = x + self.drop(self.o2(a2))
        x = x + self.drop(self.ff(self.ln3(x)))
        return x, new_sa, (k2, v2)

class Seq2Seq(nn.Module):
    def __init__(self, d_model=512, nh=8, nlayers=6, ffn=2048, dropout=0.1, pretrained=True):
        super().__init__()
        self.encoder = Encoder(d_model, pretrained, dropout)
        self.tok_emb = nn.Embedding(VOCAB, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(MAXLEN+2, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nh, ffn, dropout) for _ in range(nlayers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)
        self.d_model = d_model
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def encode(self, x):
        return self.encoder(x)
    def forward(self, x, tgt):
        mem = self.encode(x)
        B, L = tgt.shape
        pos = torch.arange(L, device=tgt.device).unsqueeze(0)
        h = self.tok_emb(tgt) + self.pos_emb(pos)
        for l in self.layers:
            h, _, _ = l(h, mem, causal=True)
        h = self.ln_f(h)
        return self.head(h)
    @torch.no_grad()
    def generate(self, x, maxlen=MAXLEN, eos=EOS, return_scores=False):
        mem = self.encode(x)
        B = x.size(0)
        device = x.device
        ys = torch.full((B,1), SOS, dtype=torch.long, device=device)
        sa = [None]*len(self.layers); ca = [None]*len(self.layers)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for i,l in enumerate(self.layers):
            kv = l.kv2(mem); k,v = kv.chunk(2,dim=-1)
            ca[i] = (l._split(k,l.nh), l._split(v,l.nh))
        scores = torch.zeros(B, device=device)
        counts = torch.zeros(B, device=device)
        min_lp = torch.full((B,), -1e9, device=device)
        for step in range(maxlen):
            h = self.tok_emb(ys[:,-1:]) + self.pos_emb(torch.tensor([[step]], device=device))
            for i,l in enumerate(self.layers):
                h, sa[i], _ = l(h, mem, sa_cache=sa[i], ca_cache=ca[i], causal=False)
            h = self.ln_f(h)
            logits = self.head(h[:,-1,:])
            lp = torch.log_softmax(logits.float(), dim=-1)
            nxt = logits.argmax(-1)
            nll = -lp.gather(1, nxt[:,None]).squeeze(1)
            active = ~done
            scores += nll*active
            counts += active.float()
            min_lp = torch.where(active, torch.maximum(min_lp, nll), min_lp)
            ys = torch.cat([ys, nxt[:,None]], dim=1)
            done |= (nxt == eos)
            if done.all(): break
        if return_scores:
            mean_lp = -(scores/counts.clamp(min=1))
            min_lp = -min_lp
            return ys, mean_lp, min_lp
        return ys

def decode_ids(ids):
    out=[]
    for i in ids:
        if i==EOS: break
        if i in (PAD,SOS): continue
        out.append(ITOS[i])
    return ''.join(out)

_RD=None
def _rdkit():
    global _RD
    if _RD is None:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        _RD=Chem
    return _RD

def smiles_to_inchi(smi, fallback=FALLBACK_INCHI):
    Chem=_rdkit()
    try:
        m=Chem.MolFromSmiles(smi)
        if m is None: return fallback
        x=Chem.MolToInchi(m)
        return x if x else fallback
    except Exception:
        return fallback

def batch_smiles_to_inchi(smis, fallback=FALLBACK_INCHI):
    Chem=_rdkit()
    out=[]
    for smi in smis:
        try:
            m=Chem.MolFromSmiles(smi)
            if m is None: out.append(fallback); continue
            x=Chem.MolToInchi(m)
            out.append(x if x else fallback)
        except Exception:
            out.append(fallback)
    return out

def load_fallback():
    try:
        return open(os.path.join(os.path.dirname(__file__),'baseline_constant.txt')).read().strip()
    except Exception:
        return FALLBACK_INCHI
