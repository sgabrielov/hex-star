from .unet_parts import *

class HexUNet(UNet):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()

        
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        # self.bilinear = bilinear

        # Overwrite these with hexagonal equivalents
        
        # self.inc = (DoubleConv(n_channels, 64))
        # self.down1 = (Down(64, 128))
        # self.down2 = (Down(128, 256))
        # self.down3 = (Down(256, 512))
        # factor = 2 if bilinear else 1
        # self.down4 = (Down(512, 1024 // factor))
        # self.up1 = (Up(1024, 512 // factor, bilinear))
        # self.up2 = (Up(512, 256 // factor, bilinear))
        # self.up3 = (Up(256, 128 // factor, bilinear))
        # self.up4 = (Up(128, 64, bilinear))
        # self.outc = (OutConv(64, n_classes))