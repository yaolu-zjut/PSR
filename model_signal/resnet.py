from __future__ import absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import os
from args import args
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm2d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def downsample_basic_block(x, planes):
    x = F.adaptive_avg_pool2d(x, (1, 1))
    #x = nn.AvgPool2d(2,2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_().cuda()
    if isinstance(x.data, torch.cuda.FloatTensor):
        # zero_pads = zero_pads.cuda()
        zero_pads = zero_pads.to(device)

    out = Variable(torch.cat([x.data, zero_pads], dim=1))

    return out

class ResNet_signal(nn.Module):

    def __init__(self, depth, dataset=args.set, cfg=None):
        super(ResNet_signal, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6  # depth=56, n=9;depth=110, n=18

        block = BasicBlock
        if cfg == None:
            cfg = [[16]*n, [32]*n, [64]*n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg = cfg

        self.inplanes = 16
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,bias=False)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[n:2*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[2*n:3*n], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        if dataset in ['radio128', 'all_radio128']:
            num_classes = 11
        elif dataset in ['radio512', 'all_radio512']:
            num_classes = 12
        elif dataset == 'radio1024':
            num_classes = 24
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes*block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = F.adaptive_avg_pool2d(x, (1, 1))
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_signal_KD(nn.Module):

    def __init__(self, depth, num_blocks, num_classes, cfg=None):
        super(ResNet_signal_KD, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6  # depth=56, n=9;depth=110, n=18

        block = BasicBlock
        if cfg == None:
            cfg = [[16]*n, [32]*n, [64]*n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg = cfg

        self.inplanes = 16
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,bias=False)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], cfg=cfg[n:2*n], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], cfg=cfg[2*n:3*n], stride=2)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes*block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = F.adaptive_avg_pool2d(x, (1, 1))
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# resnet56一共28层第一层不算，27层，1-9，10-18，19-27，对应999，那么236789，14，21222324252627，对应为617
# our保留的层为: (0, 2, 3) (6, 7, 8) (9, 14, 21), (22, 23, 24), (25, 26, 27) 保留策略，第0层必须留，其他选两个

def ResNet56_signal_KD(dataset=args.set):
    if dataset == 'radio128':
        num_classes = 11
        # k=3
        # 3，每层保留Remaining layers: (4,)(12,)(27,)
        # Remaininglayers: (0,)(4,)(12, 27)

        # 7，每层保留Remaining layers: (0, 4)(6, 12)(13, 14, 27)
        # 14，每层保留Remaining layers: (0, 1, 4)(6, 7, 12)(13, 14, 15, 16, 17, 18, 19, 27)
        # 21，每层保留Remaining layers: (0, 1, 2, 3, 4)(7, 8, 9, 10, 11)(13, 15, 16, 18, 20, 21, 22, 23, 24, 26, 27)
        # k=5
        # 7，每层保留Remaining layers: (0,)(2,)(3, 4)(10, 20)(27,)

        return ResNet_signal_KD(depth=56, num_blocks=[1, 1, 1], num_classes=num_classes)  # num_block修改
    elif dataset == 'radio512':
        num_classes = 12
        # k=3
        # 4，每层保留Remaining layers: (0,)(1, 18)(27,)
        # 7，每层保留Remaining layers: (0,)(1, 2, 18)(19, 20, 27)
        # 14，每层保留Remaining layers: (0,)(1, 2, 3, 4, 5, 6, 7, 8, 9, 18) (19, 20, 27)
        # 21，每层保留Remaining layers: (0,)(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18)(19, 20, 21, 22, 27)
        # k=5
        # 7，需要调整Remaining layers: (0,)(1, 13)(14, 18)(20,)(27,)
        # 14，需要调整Remaining layers: (0,)(1, 2, 3, 4, 13)(14, 15, 18)(19, 20)(21, 22, 27)
        return ResNet_signal_KD(depth=56, num_blocks=[1, 1, 1], num_classes=num_classes)
    elif dataset == 'radio1024':
        num_classes = 24
        # SR-init:
        #[9, 19, 0, 18, 10, 14, 16, 2, 11, 3, 15, 17, 1, 12, 20, 4, 22, 13, 6, 21, 23, 7, 8, 24, 5, 26, 25]
        # Linear_Classifier_Probes:
        # (0, 16, 18)
        # k=3
        # 3，每层保留Remaining layers: (1,)(18,)(27,)[1, 1, 1]
        # 7，每层保留Remaining layers: (0, 1)(2, 3, 18)(19, 27)[3, 1, 2]
        # 14，每层保留Remaining layers: (0, 1)(2, 3, 4, 5, 6, 18)(19, 20, 21, 22, 23, 27)[6, 1, 6]
        # 21，每层保留Remaining layers: (0, 1)(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18)(19, 20, 21, 22, 23, 27)[9, 5, 6]
        # k=5
        # 7，需要调整Remaining layers:(0,)(4,)(5, 10)(18,)(19, 27)
        # 14，需要调整Remaining layers: (0,)(1, 4)(5, 6, 10)(11, 12, 13, 18)(19, 20, 21, 27)
        # 21，需要调整Remaining layers:(0,)(1, 2, 4)(5, 6, 7, 10)(11, 12, 13, 14, 15, 18)(19, 20, 21, 22, 23, 24, 27)
        return ResNet_signal_KD(depth=56, num_blocks=[1, 1, 1], num_classes=num_classes)
    elif dataset == 'all_radio128':
        num_classes = 11
        # SR-init:
        # [0, 9, 19, 18, 10, 11, 17, 1, 12, 20, 13, 16, 14, 15, 21, 23, 22, 3, 25, 24, 2, 6, 26, 4, 7, 8, 5]
        # Linear_Classifier_Probes:
        # (0, 3, 9, 18)
        # k=3

        # 4，he:Remaining layers: (5,)(10, 12)(27,)[1, 2, 1]
        # 4，x:Remaining layers:(5,)(10, 21)(27,)[1, 1, 2]
        # 4，ramdom:Remaining layers:(5,)(10, 19)(27,)[1, 1, 2]
        # 4，lucun:Remaining layers:(5,)(16, 21)(27,)[1, 1, 2]

        # 7，Remaining layers: (3, 5)(7, 15, 18)(22, 27)[3, 2, 2]
        # 14，Remaining layers: (0, 2, 3, 4)(6, 7, 9, 12, 17)(22, 23, 25, 26, 27)[6, 2, 5]
        # 21，Remaining layers: (0, 1, 3, 4)(7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20)(21, 24, 25, 26, 27)[6, 7, 7]
        # k=4
        # 4，Remaining layers: Remaining layers: (1,)(17,)(20,)(27,)[1, 1, 2]
        # k=5
        # 7，Remaining layers: (0,)(1, 10)(11, 17)(20,)(27,)[1, 3, 2]
        # 14，Remaining layers: (0,)(2, 5, 7)(12, 14, 17)(18, 19, 20)(22, 23, 26, 27)[3, 4, 6]
        # 21，Remaining layers: (0,)(1, 2, 4, 5, 6, 8, 9)(11, 12, 14, 15, 17)(18, 19, 20)(21, 22, 24, 25, 27)[7, 6, 7]
        return ResNet_signal_KD(depth=56, num_blocks=[1, 1, 2], num_classes=num_classes)  # num_block修改
    elif dataset == 'all_radio512':
        num_classes = 12
        # SR-init:
        #[9, 0, 19, 18, 1, 10, 11, 13, 20, 12, 16, 22, 21, 17, 14, 15, 23, 3, 2, 26, 24, 25, 6, 8, 7, 5, 4]
        # Linear_Classifier_Probes:
        # (0, 9, 18)
        # k=3
        # 3，Remaining layers: (1,)(18,)(27,)[1, 1, 1]
        # 7，Remaining layers: (0, 1)(2, 3, 18)(19, 27)[3, 1, 2]
        # 14，Remaining layers: (0, 1)(2, 3, 4, 5, 6, 18)(19, 20, 21, 22, 23, 27)[6, 1, 6]
        # 21，Remaining layers: (0, 1)(2, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17)(20, 21, 23, 24, 25, 27)[7, 7, 6]
        # k=5
        # 7，Remaining layers: (1,)(2, 9)(10, 17)(19,)(27,)[3, 2, 2]
        # 14，Remaining layers: (0, 1)(2, 3, 9)(10, 11, 17)(18, 19)(20, 21, 22, 27)[4, 4, 5]
        # 21，Remaining layers: (0, 1)(2, 3, 4, 5, 6, 9)(10, 11, 12, 13, 14, 17)(18, 19)(20, 21, 22, 23, 27)[7, 7, 6]
        return ResNet_signal_KD(depth=56, num_blocks=[1, 1, 1], num_classes=num_classes)

def ResNet110_signal_KD(dataset=args.set):
    if dataset == 'radio128':
        num_classes = 11
        # K=3
        # 4:Remaining layers:(2,)(20, 43)(54,)
        # K=4
        # 5:Remaining layers: (1,)(2,)(20, 43)(54,)
        # K=5
        # 6:Remaining layers: (0,)(2,)(3,)(20, 43)(54,)
        # K=6
        # 7:Remaining layers: (0,)(2,)(3,)(20, 42)(44,)(54,)
        # K=7
        # 7:Remaining layers:(0,)(2,)(3,)(23,)(42,)(44,)(54,)
        # 14:Remaining layers:(0,)(1, 2)(3, 4)(20, 23)(24, 42)(43, 44)(45, 46, 54)
        # 27:Remaining layers:(0,)(1, 2)(3, 4, 5, 6, 7)(20, 21, 22, 23)(24, 25, 27, 31, 38, 40)
        # (43, 44)(46, 47, 48, 49, 50, 51, 53)[7, 8, 11]
        # 42:Remaining layers:(0,)(1, 2)(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18)(20, 21, 22, 23)
        # (24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 42)(43, 44)(45, 46, 47, 48, 49, 50, 51, 54)[15, 17, 11]
        return ResNet_signal_KD(depth=110, num_blocks=[2, 1, 3], num_classes=num_classes)
    elif dataset == 'radio512':
        num_classes = 12
        # K=3
        # 4:Remaining layers: (0,)(1, 36)(54,)
        # K=4
        # 4:Remaining layers: (0,)(5,)(36,)(54,)
        # K=5
        # 5:Remaining layers: (1,)(4,)(19,)(36,)(54,)
        # K=6
        # 6:Remaining layers: (1,)(4,)(19,)(36,)(38,)(54,)
        # K=7
        # 7:Remaining layers: (1,)(3,)(6,)(19,)(36,)(38,)(54,)
        # 14:Remaining layers:(0, 1)(2, 3, 4)(5, 19)(20, 21, 36)(37, 38)(40, 54)
        # 27:Remaining layers:(0, 1)(2, 3, 4)(5, 6, 7, 8, 9, 19)(20, 21, 22, 23, 24, 25, 36)(37, 38, 39)
        # (40, 41, 42, 43, 44, 54)[9, 8, 9]
        # 42:Remaining layers:(0, 1)(2, 3, 4)(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19)[14, 14, 13]
        # (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36)(37, 38, 39)(40, 41, 42, 43, 44, 45, 46, 47, 48, 54)
        return ResNet_signal_KD(depth=110, num_blocks=[1, 1, 1], num_classes=num_classes)
    elif dataset == 'radio1024':
        num_classes = 24
        # SR-init:
        #[18, 36, 37, 0, 1, 19, 2, 24, 26, 20, 25, 38, 39, 23, 22, 28, 29, 3, 21, 5, 32, 27, 45, 10, 6, 51, 48, 33, 40, 53, 43, 49, 42, 41, 31, 30, 35, 34, 8, 11, 12, 4, 14, 7, 16, 9, 13, 17, 44, 50, 52, 15, 47, 46]
        # Linear_Classifier_Probes:
        # (0, 26, 36)

        # K=3
        # 4:Remaining layers: (0,)(1, 36)(54,)[1, 1, 1]
        # K=4
        # 4:Remaining layers: (0,)(4,)(36,)(54,)[1, 1, 1]
        # K=5
        # 5:Remaining layers: (1,)(6,)(22,)(36,)(54,)[2, 2, 1]
        # K=6
        # 6:Remaining layers: (1,)(6,)(22,)(35,)(36,)(54,)[2, 3, 1]
        # K=7
        # 7:Remaining layers: (1,)(5,)(18,)(22,)(35,)(36,)(54,)[3, 3, 1]
        # 14:Remaining layers: (0, 1)(2, 5)(6, 18)(19, 22)(23, 35)(36, 37)(38, 54)[5, 5, 3]
        # 27:Remaining layers: (0, 1)(2, 3, 4, 5)(6, 7, 8, 9, 10, 11, 12, 13, 14, 18)(19, 20, 21, 22)(23, 24, 35)
        # Remaining layers: (36, 37)(38, 54)[15, 8, 3]
        # 42:Remaining layers: (0, 1)(2, 3, 4, 5)(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18)(19, 20, 21, 22)
        # Remaining layers: (23, 24, 35)(36, 37)(38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 54)[17, 8, 16]
        return ResNet_signal_KD(depth=110, num_blocks=[1, 1, 1], num_classes=num_classes)
    elif dataset == 'all_radio128':
        num_classes = 11
        # SR-init:
        # [18, 0, 36, 37, 1, 19, 2, 3, 20, 21, 5, 4, 29, 23, 8, 38, 26, 7, 39, 22, 27, 24, 10, 28, 9, 42, 46, 6, 25, 51, 33, 11, 13, 40, 44, 50, 15, 16, 30, 43, 12, 34, 35, 41, 17, 31, 49, 53, 45, 32, 47, 14, 48, 52]
        # Linear_Classifier_Probes:
        # (0, 19, 36, 37)

        # K=3
        # 4:Remaining layers: (2,)(36, 39)(54,)[1, 1, 2]
        # K=4
        # 5:Remaining layers: (1,)(2,)(33, 36)(54,)[2, 2, 1]
        # K=5
        # 5:Remaining layers: (1,)(2,)(20,)(39,)(54,)[2, 1, 2]
        # K=6
        # 6:Remaining layers: (1,)(2,)(5,)(36,)(40,)(54,)[3, 1, 2]
        # K=7
        # 7:Remaining layers: (1,)(2,)(4,)(20,)(36,)(40,)(54,)[3, 2, 2]
        # 14:Remaining layers: (0, 1)(2, 3)(4, 5)(20, 21)(22, 36)(37, 40)(41, 54)[5, 4, 4]
        # 27:Remaining layers: (0, 1)(2, 3)(4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17)(20, 21)
        # Remaining layers: (22, 29, 36)(38, 39, 40)(41, 49, 52)[15, 5, 6]
        # 42:Remaining layers: (0, 1)(2, 3)(4, 5, 6, 7, 8, 9, 10, 11, 12, 19)(20, 21)(22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36)
        # Remaining layers: (37, 38, 39, 40)(41, 42, 43, 44, 45, 46, 47, 48, 49, 54)[12, 15, 14]
        return ResNet_signal_KD(depth=110, num_blocks=[1, 1, 2], num_classes=num_classes)  # num_block修改
    elif dataset == 'all_radio512':
        num_classes = 12
        # SR-init:
        #[0, 18, 37, 36, 1, 19, 20, 38, 21, 40, 41, 5, 4, 44, 23, 22, 39, 3, 2, 50, 43, 45, 6, 25, 42, 29, 51, 52, 53, 11, 17, 24, 47, 33, 28, 31, 49, 16, 15, 26, 34, 35, 46, 48, 7, 9, 14, 13, 27, 32, 8, 10, 12, 30]
        # Linear_Classifier_Probes:
        # (0, 18, 19, 36)
        # K=3
        # 4:Remaining layers: (1,)(36,)(38,)(54,)[1, 1, 2]
        # K=4
        # 4:Remaining layers: (2,)(36,)(38,)(54,)[1, 1, 2]
        # K=5
        # 5:Remaining layers: (1,)(21,)(36,)(38,)(54,)[1, 2, 2]
        # K=6
        # 6:Remaining layers: (0,)(18,)(21,)(36,)(38,)(54,)[1, 2, 2]
        # K=7
        # 7:Remaining layers: (0,)(18,)(21,)(36,)(38,)(45,)(54,)[1, 2, 3]
        # 14:Remaining layers: (0,)(1, 18)(19, 20, 21)(22, 36)(37, 38)(39, 45)(46, 54)[2, 5, 6]
        # 27:Remaining layers: (0,)(1, 2, 18)(19, 20, 21)(22, 23, 24, 36)(37, 38)(39, 40, 41, 42, 43, 44, 45)
        # Remaining layers: (46, 47, 48, 49, 50, 51, 54)[3, 7, 16]
        # 42:Remaining layers: (0,)(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18)(19, 20, 21)
        # (22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36)(37, 38)(39, 40, 41, 42, 43, 45)(46, 47, 48, 49, 54)[14, 14, 13]
        return ResNet_signal_KD(depth=110, num_blocks=[1, 2, 1], num_classes=num_classes)

# if __name__ == '__main__':
#     net = resnet(depth=56)
#     x=Variable(torch.FloatTensor(16, 3, 32, 32))
#     y = net(x)
#     print(y.data.shape)

# data = torch.randn(10,1,2,128)
# model = resnet(depth=8)
# out = model(data)
# print(out.shape)