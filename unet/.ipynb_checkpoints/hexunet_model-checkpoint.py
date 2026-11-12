from .hexunet_parts import *
from .unet_model import UNet

class HexUNet(UNet):
    def __init__(self, n_channels, n_classes):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=False,
        )

        self.n_channels = n_channels
        self.n_classes = n_classes

        # Replace standard U-Net components with hexagonal equivalents.
        self.inc = HDoubleConv(n_channels, 64)

        self.down1 = HDown(64, 128)
        self.down2 = HDown(128, 256)
        self.down3 = HDown(256, 512)
        self.down4 = HDown(512, 1024)

        self.up1 = HUp(1024, 512)
        self.up2 = HUp(512, 256)
        self.up3 = HUp(256, 128)
        self.up4 = HUp(128, 64)

        self.outc = HOutConv(64, n_classes)

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

        return self.outc(x)