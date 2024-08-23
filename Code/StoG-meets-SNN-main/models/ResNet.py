import sys
sys.path.append('../')
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *

class conv(nn.Module):
    def __init__(self,in_plane, out_plane, kernel_size, stride,padding, bias=True):
        super(conv, self).__init__()
        self.fwd = nn.Sequential(
            nn.Conv2d(in_plane,out_plane,kernel_size=kernel_size,stride=stride,padding=padding, bias=bias),
            nn.BatchNorm2d(out_plane)
        )

    def forward(self,x):
        x = self.fwd(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, T, stride=1, shortcut=None, tau=1.0):
        super(ResidualBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.conv1 = conv(in_ch, out_ch, 3, stride, 1, bias=False)
        self.neuron1 = LIFSpike(T, tau=tau)
        self.conv2 = conv(out_ch, out_ch, 3, 1, 1, bias=False)
        self.neuron2 = LIFSpike(T, tau=tau)
        self.right = shortcut

    def forward(self, input):
        out = self.conv1(input)
        out = self.neuron1(out)
        out = self.conv2(out)
        residual = input if self.right is None else self.right(input)
        out += residual
        out = self.neuron2(out)
        return out


class ResNet17(nn.Module):
    def __init__(self, T, num_class=10, norm=None, tau=1.0):
        super(ResNet17, self).__init__()
        self.T = T
        self.tau = tau
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.pre_conv = conv(3, 64, 3, stride=1, padding=1, bias=False)
        self.neuron1 = LIFSpike(T, tau=tau)
        self.layer1 = self.make_layer(64, 64, 3, stride=2)
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = nn.Sequential()#self.make_layer(256, 512, 6, stride=2)
        self.layer4 = nn.Sequential()#self.make_layer(512, 1024, 3, stride=2)
        self.pool = nn.AvgPool2d(2,2)
        W = 32 // 2 // 2 // 2
        self.fc1 = nn.Sequential(
            nn.Linear(128*W*W, 256),
            nn.BatchNorm1d(256)
        )
        self.fc2 = nn.Linear(256, num_class)
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = conv(in_ch, out_ch, 1, stride, 0, bias=False)
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, self.T, stride, shortcut, tau=self.tau))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch, self.T, tau=self.tau))
        return nn.Sequential(*layers)

    def set_simulation_time(self, mode='bptt', *args, **kwargs):
        # self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                # module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return
    
    def forward(self, input):
        input = self.norm(input)
        input = add_dimention(input, self.T)
        input = self.merge(input)
        x = self.pre_conv(input)
        x = self.neuron1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.expand(x)
        return x


# class ResNet19(nn.Module):
    # def __init__(self, T, num_class=10, norm=None):
    #     super(ResNet19, self).__init__()
    #     self.T = T
    #     if norm is not None and isinstance(norm, tuple):
    #         self.norm = TensorNormalization(*norm)
    #     else:
    #         self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #     self.pre_conv = conv(3, 128, 3, stride=1, padding=1, bias=False)
    #     self.neuron1 = LIFSpike()
    #     self.layer1 = self.make_layer(128, 128, 3, stride=1)
    #     self.layer2 = self.make_layer(128, 256, 3, stride=2)
    #     self.layer3 = self.make_layer(256, 512, 2, stride=2)
    #     self.layer4 = nn.Sequential()#self.make_layer(512, 1024, 3, stride=2)
    #     self.pool = SeqToANNContainer(nn.AvgPool2d(2,2))
    #     W = 32 // 2 // 2 // 2
    #     self.fc1 = SeqToANNContainer(
    #         nn.Linear(512*W*W, 256),
    #         nn.BatchNorm1d(256)
    #     )
    #     self.fc2 = SeqToANNContainer(
    #         nn.Linear(256, num_class)
    #     )

    # def make_layer(self, in_ch, out_ch, block_num, stride=1):
    #     shortcut = conv(in_ch, out_ch, 1, stride, 0, bias=False)
    #     layers = []
    #     layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))
    #     for i in range(1, block_num):
    #         layers.append(ResidualBlock(out_ch, out_ch))
    #     return nn.Sequential(*layers)

    # def forward(self, input):
    #     input = self.norm(input)
    #     input = add_dimention(input, self.T)
    #     x = self.pre_conv(input)
    #     x = self.neuron1(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = self.pool(x)
    #     x = torch.flatten(x, 2)
    #     x = self.fc1(x)
    #     x = self.fc2(x)
        
    #     return x.mean(1)


if __name__ == '__main__':
    model = ResNet17(T=8)
    x = torch.rand(2,3,32,32)
    print(model(x).shape)