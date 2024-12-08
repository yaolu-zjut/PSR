import torch.nn as nn
import torch
from model_signal.builder import get_builder
import torch.nn.functional as F
from args import args

# torch.backends.cudnn.enabled = False
# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")

        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# BasicBlock }}}

# Bottleneck {{{
class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        self.conv1 = builder.conv1x1(inplanes, width)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.conv3 = builder.conv1x1(width, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


# Bottleneck }}}

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self, builder, block, layers, dataset=args.set, base_width=64):  # ////////////////////////////
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.filter = Filter(size=3, use_filter=True)

        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        self.conv1 = nn.Conv2d(
            2, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False  # ////////////////////////////
        )
        # self.conv1 = builder.conv7x7(2, 64, stride=2, first_layer=True)  # ////////////////////////////

        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if dataset in ['radio128', 'all_radio128']:
            num_classes = 11
        elif dataset in ['radio512', 'all_radio512']:
            num_classes = 12
        elif dataset == 'radio1024':
            num_classes = 24
        self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)  # ////////////////////////////
        # self.fc = builder.conv1x1(512 * block.expansion, num_classes)  # ////////////////////////////

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                self.inplanes, planes * block.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        x = self.filter(x)
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x


class Filter(nn.Module):

    def __init__(self, size, use_filter):
        super(Filter, self).__init__()
        self.size = size
        self.use_filter = use_filter
        if use_filter:
            self.filter = nn.Parameter(torch.randn(1, 2, size, size))
            # self.filter = Variable(torch.randn(1,2,size,size)).cuda(0)
            # self.filter = nn.Parameter(torch.FloatTensor(1,2,size,size))

    def forward(self, x):
        assert len(x.shape) == 3, 'len(x.shape) must equal to 3'
        length = x.shape[-1]
        # (batch_size, 2, 130)
        x = F.pad(x, (self.size // 2, self.size // 2))
        # 3 * (batch_size, 2, 130 - size + 1)
        xs = []
        for i in range(self.size):
            xs.append(x[..., i:i + length])
        # (batch_size, 2, 130 - size + 1 , size)
        xs = torch.stack(xs, dim=-1)
        xs = xs.permute(0, 2, 1, 3)  # (batch_size, 128, 2, 3) -> (batch_size, 2, 128, 3) #改动
        # (batch_size, 2, 130 - size + 1 , size) @ (1, 2, size, size)
        if self.use_filter:
            # print(xs.shape)  # (batch_size, 2, length, size)
            # print(self.filter.shape)  # (1, 2, size, size)
            xf = xs @ self.filter
            # print(xf.shape)
            # print(xs.transpose(3, 2).shape)
            return xf @ xs.transpose(3, 2)
        else:
            return xs @ xs.transpose(3, 2)


# ResNet }}}
def ResNet18(pretrained=False, **kwargs):
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet50(pretrained=False, **kwargs):
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3], **kwargs)

def ResNet50_KD(pretrained=False, **kwargs):
    #128:
    # Remaining layers: (2,)(3, 4) (13,) (14,)[2, 1, 1, 1]
    # SR-init:3, 6, 11, 12, 15,[1, 1, 2, 1]
    # RandLayer:2, 4, 5, 9, 15[1, 2, 1, 1]
    # LCP:1, 5, 8, 12, 16[1, 1, 2, 1]

    #512:
    # Remaining layers:(0,)(4,)(8,)(15,)[1, 1, 1, 1]
    # SR-init:0, 6, 9, 16[1, 1, 1, 1]
    # RandLayer:3, 5, 11, 16[1, 1, 1, 1]
    # LCP:2, 4, 8, 16[1, 1, 1, 1]

    return ResNet(get_builder(), Bottleneck, [1, 1, 1, 1], **kwargs)


def ResNet101(pretrained=False, **kwargs):
    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3], **kwargs)


def WideResNet50_2(pretrained=False, **kwargs):
    return ResNet(
        get_builder(), Bottleneck, [3, 4, 6, 3], base_width=64 * 2, **kwargs
    )


def WideResNet101_2(pretrained=False, **kwargs):
    return ResNet(
        get_builder(), Bottleneck, [3, 4, 23, 3], base_width=64 * 2, **kwargs
    )
