import torch
import torch.nn as nn


class RNAModel(nn.Module):
    def __init__(self, arch='lstm', hidden=256, layers=3, drop=0.3, n_out=5, in_dim=None, bn=True):
        super().__init__()
        self.arch = arch
        in_dim = in_dim or (4 + 3 + 7)
        self.in_dim = in_dim
        self.bn = True
        if arch == 'cnn':
            self.encoder = CNNEncoder(in_dim, hidden, drop)
        elif arch == 'gru':
            self.encoder = LSTMEncoder(in_dim, hidden, layers, drop, cell='gru', bn=bn)
        elif arch == 'transformer':
            self.encoder = TransformerEncoder(in_dim, hidden, drop)
        else:
            self.encoder = LSTMEncoder(in_dim, hidden, layers, drop, bn=bn)
        self.head = nn.Sequential(
            nn.Linear(self.enc_out, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, n_out))

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)

    @property
    def enc_out(self):
        return self.encoder.out_dim


class LSTMEncoder(nn.Module):
    def __init__(self, in_dim, hidden, layers, drop, cell='lstm', bn=True):
        super().__init__()
        def block(i, o):
            return nn.Sequential(*([nn.Conv1d(i, o, 5, padding=2)] +
                                   ([nn.BatchNorm1d(o)] if bn else []) +
                                   [nn.ELU(), nn.Dropout(drop)]))
        self.conv = nn.Sequential(block(in_dim, hidden), block(hidden, hidden))
        RNN = nn.GRU if cell == 'gru' else nn.LSTM
        self.lstm = RNN(hidden, hidden, num_layers=layers, batch_first=True,
                        bidirectional=True, dropout=drop)
        self.out_dim = hidden * 2

    def forward(self, x):
        # x: (B, L, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        return h


class CNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, drop, nblocks=6):
        super().__init__()
        self.stem = nn.Conv1d(in_dim, hidden, 5, padding=2)
        blocks = []
        for i in range(nblocks):
            d = 2 ** (i % 4 + 1)
            blocks.append(nn.Sequential(
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d),
                nn.BatchNorm1d(hidden), nn.Dropout(drop)))
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = hidden

    def forward(self, x):
        h = self.stem(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        return h.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, hidden, drop, nlayers=4, nhead=8):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pos = nn.Parameter(torch.randn(1, 256, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, nhead, hidden * 2, drop,
                                           batch_first=True, activation='gelu',
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.out_dim = hidden

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.shape[1]]
        h = self.enc(h)
        return h
