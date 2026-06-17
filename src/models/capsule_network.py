import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, capsule_dim, kernel_size=9, stride=2, padding=0):
        super().__init__()
        self.capsule_dim = capsule_dim
        self.num_capsules = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * capsule_dim,
            kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels * capsule_dim)

    def squash(self, tensor):
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm) + 1e-8)

    def forward(self, x):
        x = self.bn(self.conv(x))
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, -1, self.capsule_dim)
        return self.squash(x)


class RoutingCapsuleLayer(nn.Module):
    def __init__(self, num_capsules_in, capsule_dim_in, num_capsules_out, capsule_dim_out, num_routing=3):
        super().__init__()
        self.num_routing = num_routing
        self.num_capsules_in = num_capsules_in
        self.num_capsules_out = num_capsules_out

        self.W = nn.Parameter(
            torch.randn(1, num_capsules_in, num_capsules_out, capsule_dim_out, capsule_dim_in) * 0.01
        )

    def squash(self, tensor):
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm) + 1e-8)

    def forward(self, x):
        batch_size = x.size(0)
        x = x[:, :self.num_capsules_in]
        x = x.unsqueeze(2).unsqueeze(-1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x).squeeze(-1)

        b = torch.zeros(batch_size, self.num_capsules_in, self.num_capsules_out, 1, device=x.device)

        for iteration in range(self.num_routing):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = self.squash(s.squeeze(1))

            if iteration < self.num_routing - 1:
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1, keepdim=True)
                b = b + agreement

        return v


class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.primary_capsules = PrimaryCapsuleLayer(
            in_channels=256, out_channels=32,
            capsule_dim=8, kernel_size=9, stride=2, padding=4,
        )

        self.routing_capsules = RoutingCapsuleLayer(
            num_capsules_in=512, capsule_dim_in=8,
            num_capsules_out=num_classes, capsule_dim_out=16,
            num_routing=3,
        )

        self.reconstruction_net = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * 64 * 64),
            nn.Sigmoid(),
        )

        self.num_classes = num_classes

    def forward(self, x, y=None):
        features = self.feature_extractor(x)
        primary_caps = self.primary_capsules(features)
        digit_caps = self.routing_capsules(primary_caps)

        lengths = torch.sqrt((digit_caps ** 2).sum(dim=-1))

        if y is None:
            _, y_pred = lengths.max(dim=1)
            y_onehot = F.one_hot(y_pred, self.num_classes).float()
        else:
            y_onehot = F.one_hot(y, self.num_classes).float()

        masked = digit_caps * y_onehot.unsqueeze(-1)
        reconstruction = self.reconstruction_net(masked.view(x.size(0), -1))
        reconstruction = reconstruction.view(-1, 3, 64, 64)

        return {
            "logits": lengths,
            "probabilities": F.softmax(lengths, dim=1),
            "capsule_lengths": lengths,
            "reconstruction": reconstruction,
            "features": digit_caps.view(x.size(0), -1),
        }


class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_val=0.5):
        super().__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_val = lambda_val

    def forward(self, lengths, targets):
        targets_onehot = F.one_hot(targets, lengths.size(1)).float()
        pos_loss = targets_onehot * F.relu(self.m_pos - lengths) ** 2
        neg_loss = (1 - targets_onehot) * F.relu(lengths - self.m_neg) ** 2
        return (pos_loss + self.lambda_val * neg_loss).sum(dim=-1).mean()
