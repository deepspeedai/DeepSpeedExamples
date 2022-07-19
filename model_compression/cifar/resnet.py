from __future__ import absolute_import
'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import torch.nn as nn
import math
from copy import deepcopy

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 residual_not,
                 batch_norm_not,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if self.batch_norm_not:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual_not:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 residual_not,
                 batch_norm_not,
                 stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        if self.batch_norm_not:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if self.batch_norm_not:
            self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batch_norm_not:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual_not:
            out += residual

        out = self.relu(out)

        return out


ALPHA_ = 1


class ResNet(nn.Module):

    def __init__(self,
                 depth,
                 residual_not=True,
                 batch_norm_not=True,
                 base_channel=16,
                 num_classes=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        # block = Bottleneck if depth >=44 else BasicBlock
        block = BasicBlock

        self.base_channel = int(base_channel)
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.inplanes = self.base_channel * ALPHA_
        self.conv1 = nn.Conv2d(3,
                               self.base_channel * ALPHA_,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(self.base_channel * ALPHA_)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.base_channel * ALPHA_, n,
                                       self.residual_not, self.batch_norm_not)
        self.layer2 = self._make_layer(block,
                                       self.base_channel * 2 * ALPHA_,
                                       n,
                                       self.residual_not,
                                       self.batch_norm_not,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       self.base_channel * 4 * ALPHA_,
                                       n,
                                       self.residual_not,
                                       self.batch_norm_not,
                                       stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.base_channel * 4 * ALPHA_ * block.expansion,
                            num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    residual_not,
                    batch_norm_not,
                    stride=1):
        downsample = None
        if (stride != 1 or
                self.inplanes != planes * block.expansion) and (residual_not):
            if batch_norm_not:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride,
                              bias=False),)

        layers = nn.ModuleList()
        layers.append(
            block(self.inplanes, planes, residual_not, batch_norm_not, stride,
                  downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, residual_not, batch_norm_not))

        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        output_list = []
        x = self.conv1(x)
        if self.batch_norm_not:
            x = self.bn1(x)
        x = self.relu(x)  # 32x32
        output_list.append(x.view(x.size(0), -1))

        for layer in self.layer1:
            x = layer(x)  # 32x32
            output_list.append(x.view(x.size(0), -1))
        for layer in self.layer2:
            x = layer(x)  # 16x16
            output_list.append(x.view(x.size(0), -1))
        for layer in self.layer3:
            x = layer(x)  # 8x8
            output_list.append(x.view(x.size(0), -1))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output_list.append(x.view(x.size(0), -1))

        # return output_list, x
        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)