import torch.nn as nn
import torch
import torch.nn.functional as F
from args import args

# num_classes = 11
# ResNet {{{
class ResNet1D(nn.Module):
    def __init__(self, dataset=args.set):
        super(ResNet1D, self).__init__()
        self.conv1 = ResidualStack(1, kernel_size=(2, 3),pool_size=(2, 2),first=True)
        self.conv2 = ResidualStack(32, kernel_size=3, pool_size=2)
        self.conv3 = ResidualStack(32, kernel_size=3, pool_size=2)
        self.conv4 = ResidualStack(32, kernel_size=3, pool_size=2)
        self.conv5 = ResidualStack(32, kernel_size=3, pool_size=2)
        self.conv6 = ResidualStack(32, kernel_size=3, pool_size=2)
        if dataset in ['radio128', 'all_radio128']:
            num_classes = 11
            self.dense = nn.Linear(64, 128)
        elif dataset in ['radio512', 'all_radio512']:
            num_classes = 12
            self.dense = nn.Linear(256, 128)
        elif dataset == 'radio1024':
            num_classes = 24
            self.dense = nn.Linear(512, 128)

        self.drop = nn.Dropout(p=0.3)
        self.classfier = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.conv1(x.unsqueeze(dim=1)).squeeze(dim=2)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x).view(x.size(0),-1)
        x = self.drop(self.dense(x))
        print(x)
        x = self.classfier(x)
        return x

class ResidualStack(nn.Module):
    def __init__(self, in_channel, kernel_size, pool_size, first=False):
        super(ResidualStack, self).__init__()
        mid_channel = 32
        padding = 1
        if first:
            conv = nn.Conv2d
            pool = nn.MaxPool2d
            self.conv1 = conv(in_channel, mid_channel, kernel_size=1, padding=0, bias=False)
            self.conv2 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=(1, padding), bias=False)
            self.conv3 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=(0, padding), bias=False)
            self.conv4 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=(1, padding), bias=False)
            self.conv5 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=(0, padding), bias=False)
            self.pool = pool(kernel_size=pool_size, stride=pool_size)
        else:
            conv = nn.Conv1d
            pool = nn.MaxPool1d
            self.conv1 = conv(in_channel, mid_channel, kernel_size=1, padding=0, bias=False)
            self.conv2 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=padding, bias=False)
            self.conv3 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=padding, bias=False)
            self.conv4 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=padding, bias=False)
            self.conv5 = conv(mid_channel, mid_channel, kernel_size=kernel_size, padding=padding, bias=False)
            self.pool = pool(kernel_size=pool_size, stride=pool_size)
    def forward(self, x):
        # x = torch.transpose(x, 1, 2)
        # residual 1
        x = self.conv1(x)
        shortcut = x
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x += shortcut
        x = F.relu(x)
        # residual 2
        shortcut = x
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x += shortcut
        x = F.relu(x)
        x = self.pool(x)
        return x

class ResNet1D_KD(nn.Module):
    def __init__(self, dataset=args.set):
        super(ResNet1D_KD, self).__init__()

        if dataset =='all_radio128':
            num_classes = 11
            #Remaining layers: (0,) (4,)
            # SR-init:(0,) (3,)
            # RandLayer:(0,) (1,)
            # LCP:(0,) (5,)
            self.conv1 = ResidualStack(1, kernel_size=(2, 3), pool_size=(2, 2), first=True)
            # self.conv2 = ResidualStack(32, kernel_size=3, pool_size=2)
            # self.conv3 = ResidualStack(32, kernel_size=3, pool_size=2)
            # self.conv4 = ResidualStack(32, kernel_size=3, pool_size=2)
            # self.conv5 = ResidualStack(32, kernel_size=3, pool_size=2)
            self.conv6 = ResidualStack(32, kernel_size=3, pool_size=2)
            self.pool = nn.AdaptiveAvgPool1d(2)  # 池化到 64
            self.dense = nn.Linear(64, 128)
        elif dataset =='all_radio512':
            num_classes = 12
            # Remaining layers: (0,) (4,)
            # SR-init:(0,) (3,)
            # RandLayer:(0,) (1,)
            # LCP:(0,) (5,)
            self.conv1 = ResidualStack(1, kernel_size=(2, 3), pool_size=(2, 2), first=True)
            # self.conv2 = ResidualStack(32, kernel_size=3, pool_size=2)
            # self.conv3 = ResidualStack(32, kernel_size=3, pool_size=2)
            # self.conv4 = ResidualStack(32, kernel_size=3, pool_size=2)
            # self.conv5 = ResidualStack(32, kernel_size=3, pool_size=2)
            self.conv6 = ResidualStack(32, kernel_size=3, pool_size=2)
            self.pool = nn.AdaptiveAvgPool1d(8)  # 池化到 256
            self.dense = nn.Linear(256, 128)
        elif dataset == 'radio1024':
            num_classes = 24
            #Remaining layers: (0,) (2,)
            # SR-init:(0,) (3,)
            # RandLayer:(0,) (1,)
            # LCP:(0,) (5,)
            self.conv1 = ResidualStack(1, kernel_size=(2, 3), pool_size=(2, 2), first=True)
            # self.conv2 = ResidualStack(32, kernel_size=3, pool_size=2)
            # self.conv3 = ResidualStack(32, kernel_size=3, pool_size=2)
            # self.conv4 = ResidualStack(32, kernel_size=3, pool_size=2)
            # self.conv5 = ResidualStack(32, kernel_size=3, pool_size=2)
            self.conv6 = ResidualStack(32, kernel_size=3, pool_size=2)
            self.pool = nn.AdaptiveAvgPool1d(16)  # 池化到 256
            self.dense = nn.Linear(512, 128)

        self.drop = nn.Dropout(p=0.3)
        self.classfier = nn.Linear(128, num_classes)

    def forward(self, x):

        # x = self.conv1(x.unsqueeze(dim=1)).squeeze(dim=2)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x).view(x.size(0),-1)
        # x = self.classfier(self.drop(self.dense(x)))
        # Apply conv1
        x = self.conv1(x.unsqueeze(dim=1)).squeeze(dim=2)
        if hasattr(self, 'conv2'):
            x = self.conv2(x)
        if hasattr(self, 'conv3'):
            x = self.conv3(x)
        if hasattr(self, 'conv4'):
            x = self.conv4(x)
        if hasattr(self, 'conv5'):
            x = self.conv5(x)
        if hasattr(self, 'conv6'):
            x = self.conv6(x)
        # print(f"Shape before dense: {x.shape}")
        x = self.pool(x)  # 添加池化操作
        # print(f"Shape before dense: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"Shape before dense: {x.shape}")
        x = self.classfier(self.drop(self.dense(x)))
        return x

