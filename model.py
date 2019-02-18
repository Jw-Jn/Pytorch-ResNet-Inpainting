import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inC, outC, downsample=False, upsample=False):
        super(ResBlock, self).__init__()
        self.ds = downsample
        self.us = upsample

        if downsample:
            self.reflect_pad = nn.ReflectionPad2d(1)
            self.conv1 = nn.Conv2d(inC, outC, 3, 2, bias=False)
            self.bn = nn.BatchNorm2d(outC)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(outC, outC, 3, bias=False)
            self.conv_skip = nn.Conv2d(inC, outC, 1, 2, bias=False)
        
        elif upsample:
            self.reflect_pad = nn.ReflectionPad2d(1)
            self.deconv1 = nn.ConvTranspose2d(inC, outC, 3, 2, padding=3, output_padding=1, bias=False)
            self.bn = nn.BatchNorm2d(outC)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(outC, outC, 3, bias=False)
            self.conv_skip = nn.Conv2d(inC, outC, 1, bias=False)
            self.upsample = nn.UpsamplingNearest2d(outC)

        else:
            self.reflect_pad = nn.ReflectionPad2d(1)
            self.conv1 = nn.Conv2d(inC, outC, 3, bias=False)
            self.bn = nn.BatchNorm2d(outC)
            self.relu = nn.ReLU()

    def forward(self, x):
        if self.ds:
            # print('downsample')
            out = self.reflect_pad(x)
            # print(out.shape)
            out = self.relu(self.bn(self.conv1(out)))
            # print(out.shape)
            out = self.reflect_pad(out)
            # print(out.shape)
            out = self.bn(self.conv2(out))
            # print(out.shape)
            out = self.relu(out + self.conv_skip(x))
            # print(out.shape)

        elif self.us:
            # print('upsample')
            out = self.reflect_pad(x)
            # print(out.shape)
            out = self.relu(self.bn(self.deconv1(out)))
            # print(out.shape)
            out = self.reflect_pad(out)
            # print(out.shape)
            out = self.bn(self.conv1(out))
            # print(out.shape)
            x = self.upsample(self.conv_skip(x))
            # print(x.shape)
            out = self.relu(out + x)
            # print(out.shape)

        else:
            # print('standard')
            out = self.reflect_pad(x)
            # print(out.shape)
            out = self.relu(self.bn(self.conv1(out)))
            # print(out.shape)
            out = self.reflect_pad(out)
            # print(out.shape)
            out = self.bn(self.conv1(out))
            # print(out.shape)
            out = self.relu(out + x)
            # print(out.shape)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.reflect_pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(4, 32, 7, bias=False)
        self.reflect_pad2 = nn.ReflectionPad2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, 2, bias=False)
        self.downsample = ResBlock(64, 128, downsample=True)

        self.res_layers = nn.ModuleList([ResBlock(128, 128) for i in range(6)])

        self.upsample = ResBlock(128, 64, upsample=True)
        self.reflect_pad3 = nn.ReflectionPad2d(1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 2, padding=4, output_padding=1 ,bias=False)
        self.reflect_pad4 = nn.ReflectionPad2d(3)
        self.conv3 = nn.Conv2d(32, 3, 7)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('res-down')
        out = self.reflect_pad1(x)
        # print('r1',out.shape)
        out = self.relu(self.conv1(out))
        # print('conv1',out.shape)
        out = self.reflect_pad2(out)
        # print('r2',out.shape)
        out = self.relu(self.conv2(out))
        # print('conv2',out.shape)
        out = self.downsample(out)

        for i in range(6):
            out = self.res_layers[i](out)

        out = self.upsample(out)
        # print('res-up')
        out = self.reflect_pad3(out)
        # print('r3',out.shape)
        out = self.relu(self.deconv1(out))
        # print('deconv',out.shape)
        out = self.reflect_pad4(out)
        # print('r4',out.shape)
        out = self.conv3(out)
        # print('conv3',out.shape)

        out = self.sigmoid(out)

        return out


# self.conv = nn.Sequential()