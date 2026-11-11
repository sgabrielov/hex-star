import torch
import torch.nn as nn
import torch.nn.functional as F


class HexConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        # Axial radius-1 hex kernel:
        #
        #   x x
        # x x x
        # x x
        #
        mask = torch.tensor([
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ], dtype=torch.float32)

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