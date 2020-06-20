""" Full assembly of the parts to form the complete network """
import sys
sys.path.append("..")
import torch.nn.functional as F
import torch
import torch.nn as nn

class UNet_5(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,dropout = 0.1):
        super(UNet_5, self).__init__()
        self.cnt = 0
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout = dropout
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        if (dropout != 0): self.drop1 = torch.nn.Dropout2d(dropout)
        self.up2 = Up(512, 128, bilinear)
        if (dropout != 0): self.drop2 = torch.nn.Dropout2d(dropout)
        self.up3 = Up(256, 64, bilinear)
        if (dropout != 0): self.drop3 = torch.nn.Dropout2d(dropout)
        self.up4 = Up(128, 64, bilinear)
        if (dropout != 0): self.drop4 = torch.nn.Dropout2d(dropout)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        if (self.dropout != 0): x = self.drop1(x)
        x = self.up2(x, x3)
        if (self.dropout != 0): x = self.drop2(x)
        x = self.up3(x, x2)
        if (self.dropout != 0): x = self.drop3(x)
        x = self.up4(x, x1)
        if (self.dropout != 0): x = self.drop4(x)
        logits = self.outc(x)
        return logits

class UNet_6(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,dropout = 0.1):
        super(UNet_6, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 1024)
        self.up2 = Up(2048, 512, bilinear)
        if(dropout != 0):self.drop2 = torch.nn.Dropout2d(dropout)
        self.up3 = Up(1024, 256, bilinear)
        if (dropout != 0): self.drop3 = torch.nn.Dropout2d(dropout)
        self.up4 = Up(512, 128, bilinear)
        if (dropout != 0): self.drop4 = torch.nn.Dropout2d(dropout)
        self.up5 = Up(256, 64, bilinear)
        if (dropout != 0): self.drop5 = torch.nn.Dropout2d(dropout)
        self.up6 = Up(128, 64, bilinear)
        if (dropout != 0): self.drop6 = torch.nn.Dropout2d(dropout)

        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up2(x6, x5)
        if(self.dropout != 0):x = self.drop2(x)
        x = self.up3(x, x4)
        if (self.dropout != 0): x = self.drop3(x)
        x = self.up4(x, x3)
        if (self.dropout != 0): x = self.drop4(x)
        x = self.up5(x, x2)
        if (self.dropout != 0): x = self.drop5(x)
        x = self.up6(x, x1)
        if (self.dropout != 0): x = self.drop6(x)
        x = self.outc(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# class UNet(nn.Module):
#     if(Config.layer_numbers_unet == 4):
#         def __init__(self, n_channels, n_classes, bilinear=True, dropout=0.1):
#             super(UNet, self).__init__()
#             self.cnt = 0
#             self.n_channels = n_channels
#             self.n_classes = n_classes
#             self.dropout = dropout
#             self.bilinear = bilinear
#             self.inc = DoubleConv(n_channels, 64)
#             self.down1 = Down(64, 128)
#             self.down2 = Down(128, 256)
#             self.down3 = Down(256, 512)
#             self.down4 = Down(512, 512)
#             self.up1 = Up(1024, 256, bilinear)
#             # if (dropout != 0): self.drop1 = torch.nn.Dropout2d(dropout)
#             self.up2 = Up(512, 128, bilinear)
#             # if (dropout != 0): self.drop2 = torch.nn.Dropout2d(dropout)
#             self.up3 = Up(256, 64, bilinear)
#             # if (dropout != 0): self.drop3 = torch.nn.Dropout2d(dropout)
#             self.up4 = Up(128, 64, bilinear)
#             # if (dropout != 0): self.drop4 = torch.nn.Dropout2d(dropout)
#             self.outc = OutConv(64, n_classes)
#
#         def forward(self, x):
#             x1 = self.inc(x)
#             x2 = self.down1(x1)
#             x3 = self.down2(x2)
#             x4 = self.down3(x3)
#             x5 = self.down4(x4)
#             x = self.up1(x5, x4)
#             # if (self.dropout != 0): x = self.drop1(x)
#             x = self.up2(x, x3)
#             # if (self.dropout != 0): x = self.drop2(x)
#             x = self.up3(x, x2)
#             # if (self.dropout != 0): x = self.drop3(x)
#             x = self.up4(x, x1)
#             # if (self.dropout != 0): x = self.drop4(x)
#             logits = self.outc(x)
#             return logits
#
#     if (Config.layer_numbers_unet == 6):
#         def __init__(self, n_channels, n_classes, bilinear=True,dropout = Config.drop_rate):
#             super(UNet, self).__init__()
#             self.n_channels = n_channels
#             self.cnt = 0
#             self.n_classes = n_classes
#             self.bilinear = bilinear
#             self.dropout = dropout
#
#             self.inc = DoubleConv(n_channels, 64)
#             self.down1 = Down(64, 128)
#             self.down2 = Down(128, 256)
#             self.down3 = Down(256, 512)
#             self.down4 = Down(512, 1024)
#             self.down5 = Down(1024, 2048)
#             self.mid = Down(2048, 2048)
#             self.up1 = Up(4096, 1024, bilinear)
#             if (dropout != 0): self.drop1 = torch.nn.Dropout2d(dropout)
#             self.up2 = Up(2048, 512, bilinear)
#             if(dropout != 0):self.drop2 = torch.nn.Dropout2d(dropout)
#             self.up3 = Up(1024, 256, bilinear)
#             if (dropout != 0): self.drop3 = torch.nn.Dropout2d(dropout)
#             self.up4 = Up(512, 128, bilinear)
#             if (dropout != 0): self.drop4 = torch.nn.Dropout2d(dropout)
#             self.up5 = Up(256, 64, bilinear)
#             if (dropout != 0): self.drop5 = torch.nn.Dropout2d(dropout)
#             self.up6 = Up(128, 64, bilinear)
#             if (dropout != 0): self.drop6 = torch.nn.Dropout2d(dropout)
#
#             self.outc = OutConv(64, n_classes)
#
#         def forward(self, x):
#             x1 = self.inc(x)
#
#             x2 = self.down1(x1)
#             x3 = self.down2(x2)
#             x4 = self.down3(x3)
#             x5 = self.down4(x4)
#             x6 = self.down5(x5)
#             x7 = self.mid(x6)
#             x = self.up1(x7, x6)
#             if (self.dropout != 0): x = self.drop1(x)
#             x = self.up2(x, x5)
#             if(self.dropout != 0):x = self.drop2(x)
#             x = self.up3(x, x4)
#             if (self.dropout != 0): x = self.drop3(x)
#             x = self.up4(x, x3)
#             if (self.dropout != 0): x = self.drop4(x)
#             x = self.up5(x, x2)
#             if (self.dropout != 0): x = self.drop5(x)
#             x = self.up6(x, x1)
#             if (self.dropout != 0): x = self.drop6(x)
#
#             x = self.outc(x)
#             return x
#
#     if (Config.layer_numbers_unet == 5):
#         def __init__(self, n_channels, n_classes, bilinear=True,dropout = Config.drop_rate):
#             super(UNet, self).__init__()
#             self.n_channels = n_channels
#             self.cnt = 0
#             self.n_classes = n_classes
#             self.bilinear = bilinear
#             self.dropout = dropout
#
#             self.inc = DoubleConv(n_channels, 64)
#             self.down1 = Down(64, 128)
#             self.down2 = Down(128, 256)
#             self.down3 = Down(256, 512)
#             self.down4 = Down(512, 1024)
#             self.down5 = Down(1024, 1024)
#             # self.down6 = Down(2048, 2048)
#             # self.up1 = Up(2048, 1024, bilinear)
#             self.up2 = Up(2048, 512, bilinear)
#             if(dropout != 0):self.drop2 = torch.nn.Dropout2d(dropout)
#             self.up3 = Up(1024, 256, bilinear)
#             if (dropout != 0): self.drop3 = torch.nn.Dropout2d(dropout)
#             self.up4 = Up(512, 128, bilinear)
#             if (dropout != 0): self.drop4 = torch.nn.Dropout2d(dropout)
#             self.up5 = Up(256, 64, bilinear)
#             if (dropout != 0): self.drop5 = torch.nn.Dropout2d(dropout)
#             self.up6 = Up(128, 64, bilinear)
#             if (dropout != 0): self.drop6 = torch.nn.Dropout2d(dropout)
#
#             self.outc = OutConv(64, n_classes)
#
#
#
#         def forward(self, x):
#             x1 = self.inc(x)
#             x2 = self.down1(x1)
#             x3 = self.down2(x2)
#             x4 = self.down3(x3)
#             x5 = self.down4(x4)
#             x6 = self.down5(x5)
#             # x7 = self.down6(x6)
#             # x = self.up1(x7, x6)
#             x = self.up2(x6, x5)
#             if(self.dropout != 0):x = self.drop2(x)
#             x = self.up3(x, x4)
#             if (self.dropout != 0): x = self.drop3(x)
#             x = self.up4(x, x3)
#             if (self.dropout != 0): x = self.drop4(x)
#             x = self.up5(x, x2)
#             if (self.dropout != 0): x = self.drop5(x)
#             x = self.up6(x, x1)
#             if (self.dropout != 0): x = self.drop6(x)
#             x = self.outc(x)
#             return x