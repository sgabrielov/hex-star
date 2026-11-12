import torch
import torch.nn as nn
import torch.nn.functional as F


""" Parts of the U-Net model """

class HDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            HConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            HConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class HDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            HMaxPool2d(2),
            HDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class HMaxPool2d(nn.Module):
    """Max pooling over {self, +q, +r} on an axial hex grid."""

    def __init__(
        self,
        stride=2,
        return_indices=False,
    ):
        super().__init__()

        self.stride = stride
        self.return_indices = return_indices

    def forward(self, x):
        return hmax_pool2d(
            x,
            stride=self.stride,
            #return_indices=self.return_indices,
        )

    def extra_repr(self):
        return (
            f"stride={self.stride}, "
            f"return_indices={self.return_indices}"
        )
        

# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

class HUp(nn.Module):
    """Hex upsampling then double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = HUpsample()

        self.reduce_channels = nn.Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=1,
        )

        self.conv = HDoubleConv(
            in_channels,
            out_channels,
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x1 = self.reduce_channels(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat(
            [x2, x1],
            dim=1,
        )

        return self.conv(x)
        
class HUpsample(nn.Module):
    """Expand each coarse hex into its three corresponding fine-grid hexes."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return h_upsample2d(x)


        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.HConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        
class HConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=None,
        bias=True,
    ):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        if padding is None:
            padding = kernel_size // 2

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


        # Axial radius-1 hex kernel:
        #
        #   x x
        # x x x
        # x x
        #
        # mask = torch.tensor([
        #     [0, 1, 1],
        #     [1, 1, 1],
        #     [1, 1, 0],
        # ], dtype=torch.float32)

        # Rather than hard-coding a 3x3 mask, generate from parameter
        mask = hex_kernel_mask(kernel_size)

        self.register_buffer(
            "hex_mask",
            mask[None, None, :, :]
        )

    def forward(self, x):

        weight = self.weight * self.hex_mask

        return F.conv2d(
            x,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


#### Utilities
### Hex mask generator
### Generates a mask given a kernel size

def hex_kernel_mask(kernel_size: int) -> torch.Tensor:
    """Create an axial hex mask for an odd square kernel."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    radius = kernel_size // 2

    mask = torch.zeros(
        kernel_size,
        kernel_size,
        dtype=torch.float32,
    )

    for row in range(kernel_size):
        for col in range(kernel_size):

            q = col - radius
            r = row - radius

            distance = max(
                abs(q),
                abs(r),
                abs(q + r),
            )

            if distance <= radius:
                mask[row, col] = 1

    return mask

#### Functional utilities
### Hexagonal pooling
def hmax_pool2d(x, stride=2):
    """
    Hex max pooling over:
        self
        +q neighbor
        +r neighbor

    Assumes:
        x.shape = [N, C, H, W]
        W axis = q
        H axis = r
    """

    if x.ndim != 4:
        raise ValueError(
            "Expected input shape [N, C, H, W]"
        )

    # Self
    x_self = x

    # +q neighbor
    x_q = F.pad(
        x[:, :, :, 1:],
        (0, 1, 0, 0),
        value=float("-inf"),
    )

    # +r neighbor
    x_r = F.pad(
        x[:, :, 1:, :],
        (0, 0, 0, 1),
        value=float("-inf"),
    )

    # Aggregate the three-hex neighborhood.
    pooled = torch.maximum(
        x_self,
        torch.maximum(x_q, x_r),
    )

    # Initial coarse-grid sampling rule.
    pooled = pooled[
        :,
        :,
        ::stride,
        ::stride,
    ]

    return pooled

def h_upsample2d(x, scale_factor=2):
    """Upsample a hex feature map using nearest-neighbor replication."""

    if x.ndim != 4:
        raise ValueError(
            "Expected input shape [N, C, H, W]"
        )

    return x.repeat_interleave(
        scale_factor,
        dim=2,
    ).repeat_interleave(
        scale_factor,
        dim=3,
    )


class HOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = HConv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv(x)
