import math, torch, torch.nn as nn, torch.nn.functional as F, timm


class Encoder(nn.Module):
    def __init__(self, name='convnext_tiny.fb_in22k_ft_in1k', d_model=512, size=384, stride=32):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, features_only=True)
        c = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(c, d_model, 1)
        g = size // stride
        self.pos = nn.Parameter(torch.zeros(1, g * g, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        f = self.backbone(x)[-1]          # [B,C,h,w]
        f = self.proj(f)                  # [B,d,h,w]
        B, d, h, w = f.shape
        f = f.flatten(2).transpose(1, 2)  # [B,h*w,d]
        if h * w == self.pos.shape[1]:
            f = f + self.pos
        return f


class Attention(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        assert d % nhead == 0
        self.nhead = nhead; self.hd = d // nhead
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)

    def _split(self, t):
        B, L, _ = t.shape
        return t.view(B, L, self.nhead, self.hd).transpose(1, 2)

    def forward(self, x, ctx=None, kv_cache=None, is_causal=False):
        B, L, _ = x.shape
        src = x if ctx is None else ctx
        q = self._split(self.q(x))
        if kv_cache is not None and ctx is None:
            k = self._split(self.k(src)); v = self._split(self.v(src))
            if kv_cache.get('k') is not None:
                k = torch.cat([kv_cache['k'], k], dim=2)
                v = torch.cat([kv_cache['v'], v], dim=2)
            kv_cache['k'] = k; kv_cache['v'] = v
        else:
            k = self._split(self.k(src)); v = self._split(self.v(src))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(is_causal and L > 1))
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o(out)


class DecLayer(nn.Module):
    def __init__(self, d, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.self_attn = Attention(d, nhead)
        self.n2 = nn.LayerNorm(d); self.cross_attn = Attention(d, nhead)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dim_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, cache=None, is_causal=False):
        h = self.n1(x); h = h + self.drop(self.self_attn(h, kv_cache=cache, is_causal=is_causal)); x = h
        # cross attention reads memory entirely each call
        x = x + self.drop(self.cross_attn(self.n2(x), ctx=memory))
        x = x + self.drop(self.ff(self.n3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab, d=512, nhead=8, nlayers=6, dim_ff=2048, max_len=440, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecLayer(d, nhead, dim_ff, dropout) for _ in range(nlayers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.d = d

    def forward(self, tgt, memory, is_causal=True, caches=None, start_pos=0):
        B, L = tgt.shape
        pos = torch.arange(start_pos, start_pos + L, device=tgt.device).unsqueeze(0)
        x = self.embed(tgt) * math.sqrt(self.d) + self.pos(pos)
        x = self.drop(x)
        for i, layer in enumerate(self.layers):
            c = caches[i] if caches is not None else None
            x = layer(x, memory, cache=c, is_causal=is_causal)
        return self.head(self.norm(x))


class Img2InChI(nn.Module):
    def __init__(self, vocab, encoder_name='convnext_tiny.fb_in22k_ft_in1k',
                 d=512, nhead=8, nlayers=6, dim_ff=2048, max_len=440, size=384):
        super().__init__()
        self.encoder = Encoder(encoder_name, d, size=size)
        self.decoder = Decoder(vocab, d, nhead, nlayers, dim_ff, max_len)
        self.vocab = vocab

    def encode(self, x):
        return self.encoder(x)

    def forward(self, images, tgt):
        mem = self.encoder(images)
        return self.decoder(tgt, mem, is_causal=True)

    @torch.no_grad()
    def greedy(self, images, sos, eos, max_len=440, prefix=None):
        mem = self.encoder(images)
        B = images.shape[0]; dev = images.device
        caches = [dict(k=None, v=None) for _ in range(len(self.decoder.layers))]
        out = torch.full((B, max_len), eos, dtype=torch.long, device=dev)
        if prefix:
            cur = torch.tensor(list(prefix), dtype=torch.long, device=dev).unsqueeze(0).expand(B, -1).contiguous()
            causal = True
        else:
            cur = torch.full((B, 1), sos, dtype=torch.long, device=dev)
            causal = False
        start = 0; t = 0
        while t < max_len:
            logits = self.decoder(cur, mem, is_causal=causal, caches=caches, start_pos=start)
            nxt = logits[:, -1, :].argmax(-1)
            out[:, t] = nxt
            t += 1
            start += cur.shape[1]
            cur = nxt[:, None]; causal = False
            if bool((nxt == eos).all()):
                break
        return out
