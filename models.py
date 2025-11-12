# models.py
import torch
import torch.nn as nn

def conv_block(in_c, out_c, kernel=4, stride=2, padding=1, batchnorm=True):
    layers = [nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def deconv_block(in_c, out_c, kernel=4, stride=2, padding=1, dropout=False):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding, bias=False),
              nn.BatchNorm2d(out_c),
              nn.ReLU(inplace=True)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.down2 = conv_block(features, features*2)
        self.down3 = conv_block(features*2, features*4)
        self.down4 = conv_block(features*4, features*8)
        self.down5 = conv_block(features*8, features*8)
        self.down6 = conv_block(features*8, features*8)
        self.down7 = conv_block(features*8, features*8)
        self.bottleneck = nn.Sequential(nn.Conv2d(features*8, features*8, 4, 2, 1), nn.ReLU(True))

        self.up1 = deconv_block(features*8, features*8, dropout=True)
        self.up2 = deconv_block(features*8*2, features*8, dropout=True)
        self.up3 = deconv_block(features*8*2, features*8, dropout=True)
        self.up4 = deconv_block(features*8*2, features*8)
        self.up5 = deconv_block(features*8*2, features*4)
        self.up6 = deconv_block(features*4*2, features*2)
        self.up7 = deconv_block(features*2*2, features)
        self.final = nn.Sequential(nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bn = self.bottleneck(d7)

        u1 = self.up1(bn); u1 = torch.cat([u1, d7], dim=1)
        u2 = self.up2(u1); u2 = torch.cat([u2, d6], dim=1)
        u3 = self.up3(u2); u3 = torch.cat([u3, d5], dim=1)
        u4 = self.up4(u3); u4 = torch.cat([u4, d4], dim=1)
        u5 = self.up5(u4); u5 = torch.cat([u5, d3], dim=1)
        u6 = self.up6(u5); u6 = torch.cat([u6, d2], dim=1)
        u7 = self.up7(u6); u7 = torch.cat([u7, d1], dim=1)
        return self.final(u7)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2, True),
            conv_block(features, features*2),
            conv_block(features*2, features*4),
            conv_block(features*4, features*8, stride=1, padding=1),
            nn.Conv2d(features*8, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.net(x)
