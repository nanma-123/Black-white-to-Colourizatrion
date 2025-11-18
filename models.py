# models.py
import torch
import torch.nn as nn

# helper conv blocks
def conv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, batchnorm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def deconv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, dropout=0.0):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
              nn.BatchNorm2d(out_ch),
              nn.ReLU(inplace=True)]
    if dropout>0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, ngf=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, ngf, 4, 2, 1), nn.LeakyReLU(0.2, True)) # no bn first
        self.enc2 = conv_block(ngf, ngf*2)
        self.enc3 = conv_block(ngf*2, ngf*4)
        self.enc4 = conv_block(ngf*4, ngf*8)
        self.enc5 = conv_block(ngf*8, ngf*8)
        self.enc6 = conv_block(ngf*8, ngf*8)
        self.enc7 = conv_block(ngf*8, ngf*8)
        self.enc8 = conv_block(ngf*8, ngf*8, batchnorm=False)  # bottleneck

        # Decoder (with skip connections)
        self.dec1 = deconv_block(ngf*8, ngf*8, dropout=0.5)
        self.dec2 = deconv_block(ngf*16, ngf*8, dropout=0.5)
        self.dec3 = deconv_block(ngf*16, ngf*8, dropout=0.5)
        self.dec4 = deconv_block(ngf*16, ngf*8)
        self.dec5 = deconv_block(ngf*16, ngf*4)
        self.dec6 = deconv_block(ngf*8, ngf*2)
        self.dec7 = deconv_block(ngf*4, ngf)
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        d1 = self.dec1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        out = self.dec8(d7)
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, ndf=64):  # input: grayscale + color
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.layer2 = conv_block(ndf, ndf*2)
        self.layer3 = conv_block(ndf*2, ndf*4)
        self.layer4 = conv_block(ndf*4, ndf*8, stride=1, padding=1)  # stride=1 for PatchGAN
        self.last = nn.Conv2d(ndf*8, 1, 4, 1, 1)  # output patch

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.last(x)
        return x
