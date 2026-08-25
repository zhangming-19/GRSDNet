# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import time
import random
seed = cfg.seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.device(cfg.cuda)


class SIFBlock(nn.Module):
    def __init__(self, input_chan, out_chan):
        super(SIFBlock, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_chan, out_channels=out_chan, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU())

    def forward(self, x):
        x_pooled = self.avgpool(x)
        x1 = self.conv(x_pooled)
        x2 = self.conv(x1)
        x_concat = torch.cat([x1, x2], dim=1)
        return x_concat


class AdaptivePixelLevelAttention(nn.Module):
    def __init__(self, input_channels=1):
        super(AdaptivePixelLevelAttention, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(input_channels, input_channels // 8)
        self.fc2 = nn.Linear(input_channels // 8, input_channels)

        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # 123:(2,1344,32,80)
        batch_size, channels, height, width = x.size()
        z = self.gap(x)
        z = z.view(batch_size, channels)
        z = F.relu(self.fc1(z))  # (2,1344) -> (2,168)
        z = self.fc2(z)  # (2,168) -> (2,1344)
        z = z.view(batch_size, channels, 1, 1)  # (2,1344，1,1)
        z = self.sigmoid(z)  # (2,1344，1,1)
        x_refined = x * z
        x_refined = self.conv(x_refined)
        pixel_weights = self.sigmoid(x_refined)
        refined_x = x * pixel_weights
        output = x + refined_x

        return output


class SpatialExcitation(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attn(x)
    
class ResidualExcite(nn.Module):
    def __init__(self, scale=5.0):
        super().__init__()
        self.scale = scale

    def forward(self, x_residual):
        normed = (x_residual - x_residual.mean(dim=[2,3], keepdim=True)) / \
                 (x_residual.std(dim=[2,3], keepdim=True) + 1e-6)
        excited = torch.tanh(self.scale * normed)
        return excited * x_residual

class GuidedFilterCNN(nn.Module):
    def __init__(self, in_channels=1, guide_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + guide_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        bg = self.conv(x)
        return bg


class LocalGatedPatchAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.pool_avg = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)
        self.pool_max = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_residual):
        attn = self.pool_avg(x_residual) + self.pool_max(x_residual)
        mask = self.sigmoid(attn)
        return x_residual * mask


class GRENet(nn.Module):
    def __init__(self, need_feat='RL'):
        super().__init__()
        self.need_feat = need_feat

        self.guidefilter = GuidedFilterCNN()
        if 'L' in self.need_feat:
            self.localgate = LocalGatedPatchAttention()

    def forward(self, x):
        guide = x.detach()
        x_cat = torch.cat([x, guide], dim=1)
        B_map = self.guidefilter(x_cat)

        if self.need_feat == 'B':
            return B_map, B_map, B_map, B_map, B_map
    
        R = x - B_map

        if self.need_feat == 'R':
            return B_map, R, R, R, R
        elif self.need_feat == 'RL':
            R_l = self.localgate(R)
            return B_map, R, R, R_l, R_l


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batchsize = 2
    x = torch.randn(size=(batchsize, 1, 128, 313))
    model = GuidedFilterCNN()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    num_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Number of parameters in bytes: {num_params}")

    model = LocalGatedPatchAttention()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    num_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Number of parameters in bytes: {num_params}")

    model = GRENet()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    num_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Number of parameters in bytes: {num_params}")
