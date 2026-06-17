import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        layers = []

        if out_channels != in_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = None

        channels = in_channels
        for i in range(reps):
            if grow_first:
                inc = channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = channels if i == 0 else in_channels
                outc = in_channels if i < reps - 1 else out_channels

            if start_with_relu or i > 0:
                layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(inc, outc))
            layers.append(nn.BatchNorm2d(outc))

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride=stride, padding=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        skip = self.skip(x) if self.skip is not None else x
        return skip + self.layers(x)


class XceptionDetector(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.entry_block1 = XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False)
        self.entry_block2 = XceptionBlock(128, 256, reps=2, stride=2)
        self.entry_block3 = XceptionBlock(256, 728, reps=2, stride=2)

        self.middle_blocks = nn.Sequential(*[
            XceptionBlock(728, 728, reps=3, stride=1) for _ in range(8)
        ])

        self.exit_block = XceptionBlock(728, 1024, reps=2, stride=2, grow_first=False)

        self.exit_flow = nn.Sequential(
            SeparableConv2d(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)

        self.manipulation_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)
        x = self.middle_blocks(x)
        x = self.exit_block(x)
        x = self.exit_flow(x)

        feature_map = x
        x = self.pool(x).flatten(1)
        x = self.dropout(x)

        logits = self.fc(x)
        manipulation_type = self.manipulation_head(x)

        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "manipulation_type": manipulation_type,
            "feature_map": feature_map,
            "features": x,
        }
