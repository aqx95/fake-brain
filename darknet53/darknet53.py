"""
Darknet53 originates from YoloV3 object detection model. Read up more about YoloV3 at https://arxiv.org/abs/1804.02767

Written in Pytorch
Reference: https://github.com/developer0hye/PyTorch-Darknet53/blob/master/model.py
"""

import torch
import torch.nn as nn

# Denotes each conv layer in network
class Conv(nn.Module):
    def __init__(self, input_channel, output_channel,
                 kernel_size=3, padding=1, stride=1): #yolov3 use constant padding
        super().__init__()

        self.conv = nn.Sequential(
                        nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding, bias=False)
                        nn.BatchNorm2d(out_ch)
                        nn.LeakyReLU()
                    )

    def forward(self, x):
        return self.conv(x)


# Denotes each residual block
class ResidualBlock(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        intermed_channel = input_channel//2

        self.conv1 = Conv(input_channel, intermed_channel, kernel_size=1, padding=0)
        self.conv2 = Conv(intermed_channel, input_channel)

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.conv2(output)

        return output+residual


# Darknet
class Darknet53(nn.Module):
    def __init__(self, block, num_class):
        super().__init__()

        self.conv0 = Conv(3, 32) #256x256x32
        self.conv1  = Conv(32, 64, stride=2) #128x128x64
        self.res0 = self.block_stack(block, input_channel=64, num_blocks=1) #128x128x64
        self.conv2 = Conv(64, 128, stride=2) #64x64x128
        self.res1 = self.block_stack(block, input_channel=128, num_blocks=2) #64x64x128
        self.conv3 = Conv(128, 256, stride=2) #32x32x256
        self.res2 = self.block_stack(block, input_channel=256, num_blocks=8) #32x32x256
        self.conv4 = Conv(256, 512, stride=2) #16x16x512
        self.res3 = self.block_stack(block, input_channel=512, num_blocks=8) #16x16x512
        self.conv5 = Conv(512, 1024, stride=2) #8x8x1024
        self.res4 = self.block_stack(block, input_channel=1024, num_blocks=4) #8x8x1024

        self.pooler = nn.AdaptiveAvgPool2d((1,1))
        self.dense = nn.Linear(1024, num_class)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.res0(out)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.res2(out)
        out = self.conv4(out)
        out = self.res3(out)
        out = self.conv5(out)
        out = self.res4(out)

        out = self.pooler(out)
        out = self.dense(out)
        return out

    def block_stack(self, block, input_channel, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(input_channel))
        return nn.Sequential(*layers)
