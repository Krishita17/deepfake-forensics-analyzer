import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DCTLayer(nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size
        dct_matrix = self._build_dct_matrix(block_size)
        self.register_buffer("dct_matrix", dct_matrix)

    @staticmethod
    def _build_dct_matrix(n):
        matrix = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    matrix[k, i] = 1.0 / np.sqrt(n)
                else:
                    matrix[k, i] = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
        return matrix

    def forward(self, x):
        B, C, H, W = x.shape
        bs = self.block_size

        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H_p, W_p = x.shape

        x = x.unfold(2, bs, bs).unfold(3, bs, bs)
        x = x.contiguous().view(B, C, -1, bs, bs)

        dct_out = torch.matmul(self.dct_matrix, x)
        dct_out = torch.matmul(dct_out, self.dct_matrix.t())

        num_blocks_h = H_p // bs
        num_blocks_w = W_p // bs
        dct_out = dct_out.view(B, C, num_blocks_h, num_blocks_w, bs, bs)
        dct_out = dct_out.permute(0, 1, 4, 5, 2, 3).contiguous()
        dct_out = dct_out.view(B, C * bs * bs, num_blocks_h, num_blocks_w)

        return dct_out


class SpectralGatingUnit(nn.Module):
    def __init__(self, dim, sequence_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(sequence_len, sequence_len)
        nn.init.ones_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        residual, gate = x.chunk(2, dim=-1)
        gate = self.norm(gate)
        gate = self.proj(gate.transpose(-1, -2)).transpose(-1, -2)
        return residual * gate


class FrequencyAwareNetwork(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.dct_layer = DCTLayer(block_size=8)

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            self._make_res_block(64, 128, stride=2),
            self._make_res_block(128, 256, stride=2),
            self._make_res_block(256, 512, stride=2),
        )

        self.freq_encoder = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.phase_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(512 + 512 + 256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(512, num_classes)
        self.frequency_score_head = nn.Linear(512, 1)

    @staticmethod
    def _make_res_block(in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _compute_fft_phase(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = gray.unsqueeze(1)
        fft = torch.fft.fft2(gray)
        phase = torch.angle(fft)
        return phase.expand_as(x)

    def forward(self, x):
        spatial_feat = self.spatial_encoder(x)
        spatial_feat = F.adaptive_avg_pool2d(spatial_feat, 1).flatten(1)

        dct_out = self.dct_layer(x)
        freq_feat = self.freq_encoder(dct_out)
        freq_feat = F.adaptive_avg_pool2d(freq_feat, 1).flatten(1)

        phase = self._compute_fft_phase(x)
        phase_feat = self.phase_analyzer(phase)

        combined = torch.cat([spatial_feat, freq_feat, phase_feat], dim=1)
        fused = self.fusion(combined)

        logits = self.classifier(fused)
        freq_score = torch.sigmoid(self.frequency_score_head(fused)).squeeze(-1)

        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "frequency_anomaly_score": freq_score,
            "features": fused,
        }
