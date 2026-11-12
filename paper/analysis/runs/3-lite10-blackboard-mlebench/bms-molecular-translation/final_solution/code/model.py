import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Encoder(nn.Module):
    def __init__(self, name, out_index, d_model, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            name, features_only=True, out_indices=(out_index,), pretrained=pretrained)
        ch = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(ch, d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        f = self.backbone(x)[-1]
        f = self.proj(f)
        B, d, H, W = f.shape
        f = f.flatten(2).transpose(1, 2)
        return self.norm(f), H, W


class DecLayer(nn.Module):
    def __init__(self, d, nhead, dim_ff, dropout):
        super().__init__()
        self.d = d
        self.nhead = nhead
        self.dh = d // nhead
        self.ln1 = nn.LayerNorm(d)
        self.self_qkv = nn.Linear(d, 3 * d)
        self.self_o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.cross_q = nn.Linear(d, d)
        self.cross_kv = nn.Linear(d, 2 * d)
        self.cross_o = nn.Linear(d, d)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, dim_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(dim_ff, d))
        self.drop = nn.Dropout(dropout)

    def _split(self, t, n):
        B, T, _ = t.shape
        return t.view(B, T, n, self.nhead, self.dh).permute(2, 0, 3, 1, 4)

    def forward(self, x, mem):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self._split(self.self_qkv(h), 3)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.self_o(a))
        h = self.ln2(x)
        q = self.cross_q(h).view(B, T, self.nhead, self.dh).transpose(1, 2)
        ck, cv = self._split(self.cross_kv(mem), 2)
        a = F.scaled_dot_product_attention(q, ck, cv)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.cross_o(a))
        x = x + self.drop(self.ff(self.ln3(x)))
        return x

    def cross_cache(self, mem):
        return self._split(self.cross_kv(mem), 2)

    def step(self, x, cross, cache):
        B, _, D = x.shape
        h = self.ln1(x)
        q, k, v = self._split(self.self_qkv(h), 3)
        if cache is not None:
            k = torch.cat([cache[0], k], 2)
            v = torch.cat([cache[1], v], 2)
        new_cache = (k, v)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, 1, D)
        x = x + self.drop(self.self_o(a))
        h = self.ln2(x)
        q = self.cross_q(h).view(B, 1, self.nhead, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, cross[0], cross[1])
        a = a.transpose(1, 2).reshape(B, 1, D)
        x = x + self.drop(self.cross_o(a))
        x = x + self.drop(self.ff(self.ln3(x)))
        return x, new_cache


class Model(nn.Module):
    def __init__(self, vocab, enc_name='convnext_tiny.fb_in22k_ft_in1k', out_index=2,
                 d_model=512, nhead=8, layers=6, dim_ff=2048, dropout=0.1,
                 max_len=320, pretrained=True):
        super().__init__()
        self.encoder = Encoder(enc_name, out_index, d_model, pretrained)
        self.grid = None
        self.pos = None
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(vocab, d_model)
        self.tgt_pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([DecLayer(d_model, nhead, dim_ff, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm = nn.Linear(d_model, vocab, bias=False)
        self.max_len = max_len
        self.vocab = vocab
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def _enc_pos(self, H, W, device):
        if self.pos is None or self.grid != (H, W) or self.pos.device != device:
            pe = torch.zeros(H * W, self.d_model, device=device)
            pos = torch.arange(H * W, device=device, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, self.d_model, 2, device=device, dtype=torch.float)
                            * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.pos = pe.unsqueeze(0)
            self.grid = (H, W)
        return self.pos

    def encode(self, images):
        mem, H, W = self.encoder(images)
        return mem + self._enc_pos(H, W, mem.device)

    def forward(self, images, tgt_in):
        mem = self.encode(images)
        B, T = tgt_in.shape
        pos = torch.arange(T, device=tgt_in.device)
        x = self.tgt_emb(tgt_in) * math.sqrt(self.d_model) + self.tgt_pos(pos)[None]
        for layer in self.layers:
            x = layer(x, mem)
        x = self.ln_f(x)
        return self.lm(x)

    @torch.no_grad()
    def generate_fast(self, images, bos, eos, max_len=320):
        """Same greedy decode as generate() but with preallocated KV caches
        (avoids per-step torch.cat => much faster). Call under autocast(bf16)."""
        mem = self.encode(images)
        B = mem.shape[0]
        dev = mem.device
        max_len = min(max_len, self.max_len)
        cross = [layer.cross_cache(mem) for layer in self.layers]
        L0 = self.layers[0]
        H, dh, D = L0.nhead, L0.dh, self.d_model
        cdt = cross[0][0].dtype
        sks = [torch.zeros(B, H, max_len, dh, device=dev, dtype=cdt) for _ in self.layers]
        svs = [torch.zeros(B, H, max_len, dh, device=dev, dtype=cdt) for _ in self.layers]
        ids = torch.full((B,), bos, device=dev, dtype=torch.long)
        out_tokens = torch.full((B, max_len), eos, dtype=torch.long, device=dev)
        lens = torch.zeros(B, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(max_len - 1):
            pos = torch.full((B,), t, device=dev, dtype=torch.long)
            x = (self.tgt_emb(ids) * math.sqrt(self.d_model) + self.tgt_pos(pos))[:, None, :]
            for i, layer in enumerate(self.layers):
                h = layer.ln1(x)
                q, k, v = layer._split(layer.self_qkv(h), 3)
                sks[i][:, :, t:t + 1] = k
                svs[i][:, :, t:t + 1] = v
                a = F.scaled_dot_product_attention(q, sks[i][:, :, :t + 1], svs[i][:, :, :t + 1])
                a = a.transpose(1, 2).reshape(B, 1, D)
                x = x + layer.drop(layer.self_o(a))
                h = layer.ln2(x)
                q2 = layer.cross_q(h).view(B, 1, H, dh).transpose(1, 2)
                a = F.scaled_dot_product_attention(q2, cross[i][0], cross[i][1])
                a = a.transpose(1, 2).reshape(B, 1, D)
                x = x + layer.drop(layer.cross_o(a))
                x = x + layer.drop(layer.ff(layer.ln3(x)))
            nxt = self.lm(self.ln_f(x))[:, 0].argmax(-1)
            active = ~done
            out_tokens[active, lens[active]] = nxt[active]
            lens[active] += 1
            done = done | (nxt == eos)
            ids = nxt
            if bool(done.all()):
                break
        outs = []
        for b in range(B):
            L = int(lens[b])
            toks = out_tokens[b, :L].tolist()
            if toks and toks[-1] == eos:
                toks = toks[:-1]
            outs.append(toks)
        return outs

    @torch.no_grad()
    def generate(self, images, bos, eos, max_len=320, batch_decode=True):
        mem = self.encode(images)
        B = mem.shape[0]
        cross = [layer.cross_cache(mem) for layer in self.layers]
        caches = [None] * len(self.layers)
        ids = torch.full((B, 1), bos, device=mem.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=mem.device)
        outs = [[] for _ in range(B)]
        for t in range(max_len - 1):
            pos = torch.full((B,), t, device=mem.device, dtype=torch.long)
            x = self.tgt_emb(ids[:, -1:]) * math.sqrt(self.d_model) + self.tgt_pos(pos)[:, None]
            for i, layer in enumerate(self.layers):
                x, caches[i] = layer.step(x, cross[i], caches[i])
            logits = self.lm(self.ln_f(x))[:, -1]
            nxt = logits.argmax(-1)
            for b in range(B):
                if not done[b]:
                    tok = int(nxt[b])
                    if tok == eos:
                        done[b] = True
                    else:
                        outs[b].append(tok)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            if done.all():
                break
        return outs
