import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inC, outC, downsample=False, upsample=False):
        super(ResBlock, self).__init__()
        self.ds = downsample
        self.us = upsample

        if downsample:
            self.ds_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(inC, outC, 3, 2, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(outC, outC, 3, bias=False),
            nn.BatchNorm2d(outC))
            self.conv_skip = nn.Conv2d(inC, outC, 1, 2, bias=False)
        
        elif upsample:
            self.us_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(inC, outC, 3, 2, padding=3, output_padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(outC, outC, 3, bias=False),
            nn.BatchNorm2d(outC))
            self.conv_skip = nn.Conv2d(inC, outC, 1, bias=False)
            self.upsample = nn.UpsamplingNearest2d(outC))

        else:
            self.std_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(inC, outC, 3, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inC, outC, 3, bias=False),
            nn.BatchNorm2d(outC))

    def forward(self, x):
        if self.ds:
            # print('downsample')
            out = self.ds_block(x)
            out = F.relu(out + self.conv_skip(x))
            # print(out.shape)

        elif self.us:
            # print('upsample')
            out = self.us_block(x)
            x = self.upsample(self.conv_skip(x))
            out = F.relu(out + x)
            # print(out.shape)

        else:
            # print('standard')
            out = self.std_block(x)
            out = F.relu(out + x)
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

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.reflect_pad1(x)
        out = F.relu(self.conv1(out))
        out = self.reflect_pad2(out)
        out = F.relu(self.conv2(out))
        out = self.downsample(out)

        for i in range(6):
            out = self.res_layers[i](out)

        out = self.upsample(out)
        out = self.reflect_pad3(out)
        out = F.relu(self.deconv1(out))
        out = self.reflect_pad4(out)
        out = self.conv3(out)

        out = self.sigmoid(out)

        return out


# self.conv = nn.Sequential()