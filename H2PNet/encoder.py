'''
# @Time    : 2025/5/14 9:34
# @Author  : Fengcheng Ji
# @Function: 卷积块编码器，多尺度特征融合 + 通道注意力
'''
import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32 + 64 + 64, (32 + 64 + 64) // 4),
            nn.ReLU(),
            nn.Linear((32 + 64 + 64) // 4, 32 + 64 + 64),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32 + 64 + 64, embed_dim)

    def forward(self, x):
        x1 = self.conv1(x)   # [B, 32, H, W]
        x2 = self.conv2(x1)  # [B, 64, H, W]
        x3 = self.conv3(x2)  # [B, 64, H, W]

        x_concat = torch.cat((x1, x2, x3), dim=1)  # [B, 160, H, W]
        attn_weights = self.attention(x_concat)     # [B, 160]

        w1 = attn_weights[:, :32].view(-1, 32, 1, 1)
        w2 = attn_weights[:, 32:96].view(-1, 64, 1, 1)
        w3 = attn_weights[:, 96:].view(-1, 64, 1, 1)

        x_fused = torch.cat((x1 * w1, x2 * w2, x3 * w3), dim=1)
        x_pooled = self.pool(x_fused).flatten(1)
        out = self.fc(x_pooled)
        return out
