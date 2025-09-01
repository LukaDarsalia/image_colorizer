import torch.nn as nn
from torch.nn.utils import spectral_norm

def conv_block(in_ch, out_ch, stride, spectral=False, use_groupnorm=True):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not spectral)
    if spectral:
        conv = spectral_norm(conv)

    layers = [conv]

    if use_groupnorm:
        # Equivalent to LayerNorm across channels
        layers += [nn.GroupNorm(32, out_ch)]

    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)

class PatchGAN70(nn.Module):
    def __init__(self, in_channels=3, base=64, spectral=False):
        super().__init__()
        self.head = conv_block(in_channels, base,   stride=2, spectral=spectral)  # C64
        self.b2   = conv_block(base,       base*2,  stride=2, spectral=spectral)  # C128
        self.b3   = conv_block(base*2,     base*4,  stride=2, spectral=spectral)  # C256
        self.b4   = conv_block(base*4,     base*8,  stride=1, spectral=spectral)  # C512, stride 1
        # final 1-channel conv, stride 1, no norm/activation
        self.tail = nn.Conv2d(base*8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.tail(x)  # logits map

class PatchGAN34(nn.Module):
    def __init__(self, in_channels=3, base=64, spectral=False):
        super().__init__()
        self.head = conv_block(in_channels, base,   stride=2, spectral=spectral)  # 64->32
        self.b2   = conv_block(base,       base*2,  stride=2, spectral=spectral)  # 32->16
        self.b3   = conv_block(base*2,     base*4,  stride=1, spectral=spectral)  # 16->15
        self.b4   = conv_block(base*4,     base*8,  stride=1, spectral=spectral)  # 15->14
        self.tail = nn.Conv2d(base*8, 1, kernel_size=4, stride=1, padding=1)      # 14->13

    def forward(self, x):
        x = self.head(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        return self.tail(x)