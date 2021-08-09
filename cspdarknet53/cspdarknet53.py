"""
CSPDarknet53 originates from YoloV4 object detection model. Read up more about YoloV4 at https://arxiv.org/abs/2004.10934

Written in Pytorch
Reference:
https://github.com/njustczr/cspdarknet53/blob/master/cspdarknet53/csdarknet53.py
https://github.com/romulus0914/YOLOv4-PyTorch/blob/master/CSPDarknet53.py
"""

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=1, stride=1):
        super.__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Mish()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, downsample=False):
        super().__init__()
        self.intermed_channel = in_channel
        if downsample == True:
            self.intermed_channel = in_channel // 2
        self.conv1 = Conv(in_channel, intermed_channel, kernel_size=1, padding=0)
        self.conv2 = Conv(intermed_channel, in_channel, kernel_size=3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return residual + out


class CSPBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_resblocks, downsample=False):
        super.__init__()
        self.conv = Conv(in_channel, out_channel, kernel_size=1, padding=0)
        layers = []
        for i in range(num_resblock):
            layers.append(ResidualBlock(out_channel, downsample))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.blocks(out)
        return out


# First CSP stage
class CSPStage1(nn.Module):
    def __init__(self, in_channel, num_blocks=1):
        super.__init__()
        self.conv1 = Conv(in_channel, 2*in_channel, 3, stride=2, padding=0)
        self.csp = CSPBlock(in_channel, in_channel, num_blocks, downsample=True)
        self.conv2 = Conv(2*in_channel, 2*in_channel, 1, padding=0)
        self.conv3 = Conv(2*in_channel, 2*in_channel, 1, padding=0)
        self.conv4 = Conv(4*in_channel, 2*in_channel, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        split0 = self.conv2(x)
        split1 = self.csp(x)
        split1 = self.conv3(split1)
        out = torch.cat([split0, split1], dim=1)
        out = self.conv4(out)
        return out


# Remaining CSP stages
class CSPStage(nn.Module):
    def __init__(self, in_channel, num_blocks):
        super.__init__()
        self.conv1 = Conv(in_channel, 2*in_channel, 3, stride=2, padding=0)
        self.csp = CSPBlock(2*in_channel, in_channel, num_blocks, downsample=False)
        self.conv2 = Conv(2*in_channel, in_channel, 1, padding=0)
        self.conv3 = Conv(in_channel, in_channel, 1, padding=0)
        self.conv4 = Conv(2*in_channel, 2*in_channel, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        split0 = self.conv2(x)
        split1 = self.csp(x)
        split1 = self.conv3(split1)
        out = torch.cat([split0, split1], dim=1)
        out = self.conv4(out)
        return out


# CSP-Darknet53
class CSP_Darknet53(nn.Module):
    def __init__(self, num_class):
        super.__init__()
        self.conv_layer = Conv(3, 32, 3)
        self.csp1 = CSPStage1(in_channel=32)
        self.csp2 = CSPStage(in_channel=64, num_blocks=2)
        self.csp3 = CSPStage(in_channel=128, num_blocks=8)
        self.csp4 = CSPStage(in_channel=256, num_blocks=8)
        self.csp5 = CSPStage(in_channel=512, num_blocks=4)
        self.pooler = nn.AdaptiveAvgPool2d((1,1))
        self.dense = nn.Linear(1024, num_class)

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.csp1(out)
        out = self.csp2(out)
        out = self.csp3(out)
        out = self.csp4(out)
        out = self.csp5(out)
        pooled_out = self.pooler(out)
        out = self.dense(pooled_out)
        return out
