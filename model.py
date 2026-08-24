import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
torch.cuda.device(cfg.cuda)

### model structures: mobilenetv2, Resnet32, CNN, CNNE, CNNL


### mobilenetv2
# __all__ = ['mobilenetv2']
def _make_divisible(v, divisor, min_value=None):  # 32 8
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):  # model size 8.515MB
    def __init__(self, num_classes=cfg.class_num, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(1, input_channel, 2)]  # layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult,
                                         4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280  # 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if cfg.premodel_ext is True:
            x1 = self.classifier(x)
            return x1, x
        else:
            x = self.classifier(x)
            # x = F.softmax(x)
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 models
    """
    return MobileNetV2(**kwargs)


### ResNet32
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes=cfg.class_num):
        super(ResNet32, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 3)  # Updated number of blocks
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)  # Updated number of blocks
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)  # Updated number of blocks
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)  # Updated number of blocks

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if cfg.premodel_ext is True:
            x1 = self.fc(x)
            return x1, x
        else:
            x = self.fc(x)
            # x = F.softmax(x)
            return x


### CNN
class CNN(nn.Module):
    def __init__(self, num_classes=cfg.class_num):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 32, kernel_size=7, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 20, kernel_size=7, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_layers = nn.Linear(20 * 13 * 55, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        if cfg.premodel_ext is True:
            x1 = self.fc_layers(x)
            return x1, x
        else:
            x = self.fc_layers(x)
            return x


### CNNE
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class CNNE(nn.Module):
    def __init__(self, num_classes=cfg.class_num):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.layer1 = nn.Sequential(
            BasicBlock(32, 32, stride=(1, 1)),
            BasicBlock(32, 32, stride=(1, 1)),
            BasicBlock(32, 32, stride=(1, 1)),
            BasicBlock(32, 64, stride=(2, 2)))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 64, stride=(1, 1)),
            BasicBlock(64, 64, stride=(1, 1)),
            BasicBlock(64, 64, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(64, 128, stride=(2, 2)))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 128, stride=(1, 1)),
            BasicBlock(128, 128, stride=(1, 1)),
            BasicBlock(128, 128, stride=(1, 1)),
            BasicBlock(128, 256, stride=(2, 2)), )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 256, stride=(1, 1)),
            BasicBlock(256, 256, stride=(1, 1)),
            BasicBlock(256, 256, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(256, 512, stride=(2, 2)))

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if cfg.premodel_ext is True:
            x1 = self.classifier(x)
            return x1, x
        else:
            x = self.classifier(x)
            return x


### CNNL
class CNNL(nn.Module):
    def __init__(self, num_classes=cfg.class_num):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer1 = nn.Sequential(
            BasicBlock(32, 32, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(32, 64, stride=(2, 2)))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 64, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(64, 128, stride=(2, 2)))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 128, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(128, 256, stride=(2, 2)), )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 256, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(256, 512, stride=(2, 2)))

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.layer0(x)  # 32 64 234
        x = self.layer1(x)  #
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 512 4 14
        x = self.avgpool(x)  # 512 1 1
        x = x.view(x.size(0), -1)
        if cfg.premodel_ext is True:
            x1 = self.classifier(x)
            return x1, x
        else:
            x = self.classifier(x)  # 5
            return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batchsize = 2
    x = torch.randn(size=(batchsize, 1, 128, 469))
    model = CNNL()
    output_z = model(x)  # , output_recon
    print(f'input dim:{x.shape}. output_z dim:{output_z[1].shape}.')  # output_recon dim:{output_recon.shape}
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the models: {num_params}")
    num_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Number of parameters in bytes: {num_params}")
