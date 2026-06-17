import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .attention_module import MultiHeadSelfAttention, SpatialAttentionBlock


class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64):
        super().__init__()
        self.dct_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.high_pass_kernel = nn.Parameter(
            torch.tensor([
                [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            ], dtype=torch.float32).unsqueeze(0).repeat(in_channels, 1, 1, 1),
            requires_grad=False,
        )
        self.fusion = nn.Conv2d(mid_channels + in_channels, mid_channels, kernel_size=1)

    def forward(self, x):
        high_freq = F.conv2d(x, self.high_pass_kernel, padding=1, groups=x.size(1))
        dct_features = self.dct_conv(x)
        combined = torch.cat([dct_features, high_freq], dim=1)
        return self.fusion(combined)


class SRMFilterLayer(nn.Module):
    def __init__(self):
        super().__init__()
        srm_filters = torch.zeros(3, 1, 5, 5)
        srm_filters[0, 0] = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32)
        srm_filters[1, 0] = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -2, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32)
        srm_filters[2, 0] = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32) / 4.0
        self.srm = nn.Parameter(srm_filters, requires_grad=False)

    def forward(self, x):
        channels = []
        for c in range(x.size(1)):
            channels.append(F.conv2d(x[:, c:c+1], self.srm, padding=2))
        return torch.cat(channels, dim=1)


class EfficientNetDetector(nn.Module):
    def __init__(self, model_name="efficientnet_b4", num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features

        self.srm_layer = SRMFilterLayer()
        self.freq_branch = FrequencyBranch(in_channels=3, mid_channels=64)

        self.spatial_attention = SpatialAttentionBlock(feature_dim)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 16, feature_dim),
            nn.Sigmoid(),
        )

        self.srm_projector = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
        )

        self.freq_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim + 256 + 256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(512, num_classes)
        self.anomaly_head = nn.Linear(512, 1)

    def extract_features(self, x):
        backbone_feat = self.backbone(x)
        return backbone_feat

    def forward(self, x):
        backbone_feat = self.backbone(x)

        srm_out = self.srm_layer(x)
        srm_feat = self.srm_projector(srm_out)

        freq_out = self.freq_branch(x)
        freq_feat = self.freq_projector(freq_out)

        fused = torch.cat([backbone_feat, srm_feat, freq_feat], dim=1)
        fused = self.fusion_layer(fused)

        logits = self.classifier(fused)
        anomaly_score = self.anomaly_head(fused).squeeze(-1)

        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "anomaly_score": torch.sigmoid(anomaly_score),
            "features": fused,
        }
