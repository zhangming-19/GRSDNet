import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
torch.cuda.device(cfg.cuda)


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

# original mobilefacenet setting
# Mobilefacenet_bottleneck_setting = [
#     # t, c , n ,s
#     [2, 64, 5, 2],
#     [4, 128, 1, 2],
#     [2, 128, 6, 1],
#     [4, 128, 1, 2],
#     [2, 128, 2, 1]
# ]

# Mobilenetv2_bottleneck_setting = [
#     # t, c, n, s
#     [1, 16, 1, 1],
#     [6, 24, 2, 2],
#     [6, 32, 3, 2],
#     [6, 64, 4, 2],
#     [6, 96, 3, 1],
#     [6, 160, 3, 2],
#     [6, 320, 1, 1],
# ]

# refer to DCASE2022 Task2 Top-1
# https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Liu_8_t2.pdf
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]


class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class=cfg.class_num,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 premodel_ext=cfg.premodel_ext,
                 learn_method=cfg.learn_method, input_dim=1, attention_choose=False):
        super(MobileFaceNet, self).__init__()

        self.premodel_ext = premodel_ext
        self.learn_method = learn_method
        self.attention_choose = attention_choose
        self.input_dim = input_dim

        self.conv1 = ConvBlock(self.input_dim, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)

        if self.attention_choose == 'SIF':
            linear7_input_size = 1024
            self.linear7 = ConvBlock(linear7_input_size, 512, (8, 20), 1, 0, dw=False, linear=True)
        else:
            linear7_input_size = 512
            self.linear7 = ConvBlock(linear7_input_size, 512, (8, 20), 1, 0, dw=True, linear=True)

        # self.linear7 = ConvBlock(512, 512, (4, 10), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        fc_out_input_size = 256 if self.attention_choose == 'SIF' else 128

        self.fc_out = nn.Linear(128, num_class)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)  # B,512,8,20
        x = self.linear7(x)
        x = self.linear1(x)

        feature = x.view(x.size(0), -1)

        if (self.premodel_ext == 'True') or self.learn_method == 'Contrastive':
            out = self.fc_out(feature)
            return out, feature  # B,128
        else:
            out = self.fc_out(feature)
            return out
