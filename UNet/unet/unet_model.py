""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, tanh=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.tanh = tanh

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        #self.down5 = Down(1024 // factor, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.tanh:
            logits = self.tanh(logits)
        return logits

class UNetHalf(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetHalf, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.down5 = Down(1024 // factor, 1024 // factor)
        self.avgPool = torch.nn.AdaptiveAvgPool2d(1)
        self.outc = OutConv(1024 // factor, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.avgPool(x6)
        x8 = self.outc(x7)
        return self.activation(x8)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False,
                                  act=False)]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True,
                   act=True):  # when needing to make some repeatitive blocks of layers,
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)]  # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = False

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownNoPool(64, 128, nn.LeakyReLU())
        self.down2 = DownNoPool(128, 256, nn.LeakyReLU())
        self.down3 = DownNoPool(256, 512, nn.LeakyReLU())
        factor = 2 if bilinear else 1
        self.down4 = DownNoPool(512, 1024 // factor, nn.LeakyReLU())
        #self.down5 = Down(1024 // factor, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear, nn.ReLU())
        self.up2 = Up(512, 256 // factor, bilinear, nn.ReLU())
        self.up3 = Up(256, 128 // factor, bilinear, nn.ReLU())
        self.up4 = Up(128, 64, bilinear, nn.ReLU())
        #self.outc = OutConv(64, n_classes) # Remove this FC layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, bias=False)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.tanh(logits)

class UNet2Half(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2Half, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = False
        activation = nn.LeakyReLU()

        self.inc = DoubleConv(n_channels, 64, activation=activation)
        self.down1 = DownNoPool(64, 128, activation=activation)
        self.down2 = DownNoPool(128, 256, activation=activation)
        self.down3 = DownNoPool(256, 512, activation=activation)
        factor = 2 if bilinear else 1
        self.down4 = DownNoPool(512, 1024 // factor, activation=activation)
        #self.down5 = DownNoPool(1024 // factor, 1024 // factor, activation=activation)
        self.down5 = DownNoPool(1024 // factor, 1, activation=activation)
        #self.avgPool = torch.nn.AdaptiveAvgPool2d(1)
        #self.outc = OutConv(1024 // factor, 1)
        self.flatten = nn.Flatten()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        #x7 = self.avgPool(x6)
        #x8 = self.outc(x7)
        x8 = self.flatten(x6)
        return self.activation(x8)