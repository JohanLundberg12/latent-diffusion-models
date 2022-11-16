"""Based on Deep Residual Learning for Image Recognition paper:
https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
& code based on: https://nn.labml.ai/resnet/index.html
"""
from typing import List, Optional

import torch
from torch import nn


# A typical residucal connection is F(x) + x, also written as
# F(x, {W_i}) + x
# Authors of the paper suggest doing a linear project
# (W_s x) when F(x, W_i) and x are different: F(x, {W_i}) + W_s x.
# It ups the number of filters to match the output of F with x.
class ShortcutProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # 1x1 convolution is what the world has ended up doing
        # the paper experimented with other choices, s.a. 0 padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


# The residual block as describes in the paper. It has two 3x3 conv layers.
# A conv layer is followed by batchnorm, thus bias=False in the conv layers, as
# the bias is cancelled out anyway by the batchnorm.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """Residual block with relu((conv, norm, relu, conv, norm) + residual)
        Args:
            in_channels (int): number of channels in x.
            out_channels (int): number of output channels.
            stride (int): stride length in the convolution.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If stride is not 1, we need a shortcut projection to change the number of channels
        if stride != 1 or in_channels != out_channels:
            # W_s x
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # x is the input of shape [batch_size, in_channels, height, width]

        # First the shortcut connection (residual)
        shortcut = self.shortcut(x)

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # Activation(x + residual)
        return self.act2(x + shortcut)


# An implementation of the bottleneck block described in the paper.
# It has a 1x1, a 3x3 and a 1x1 conv layer, each followed by a batchnorm and relu.
# It also has a residual connection.
class BottleneckResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int
    ):
        """Bottlenect residual block with
        relu((conv, norm, relu, conv, norm, relu, conv, norm) + res)
        Args:
            in_channels (int): number of channels in x.
            bottleneck_channels (int): number of channels for the 3x3 conv.
            out_channels (int): number of output channels.
            stride (int): stride length for the 3x3 conv.
        """
        super().__init__()

        # First 1×1 convolution layer, this maps to bottleneck_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU()

        # Third 1×1 convolution layer, this maps to out_channels.
        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut connection should be a projection if the stride length is not 1
        # or if the number of channels change.
        if stride != 1 or in_channels != out_channels:
            # W_s x -> it will expand the number of filters using 1x1 conv
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

        # Third activation function (ReLU) (after adding the shortcut)
        self.act3 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # x is the input of shape [batch_size, in_channels, height, width]

        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return self.act3(x + shortcut)


# This is a the base of the resnet model
# without the final linear layer and softmax for classification.
#
# The resnet is made of stacked residual blocks or bottleneck residual blocks.
# The feature map size is halved after a few blocks with a block of stride length 2.
# The number of channels is increased when the feature map size is reduced.
# Finally the feature map is average pooled to get a vector representation.
# n_blocks[0] = 2 means two resnet blocks for the feature mapping with 64 channels
# also, this one won't have any downscaling as this was done in the initial conv layer.
class ResNetBase(nn.Module):
    def __init__(
        self,
        img_channels: int,
        out_channels: int,
        n_blocks: List[int],
        n_channels: List[int],
        bottlenecks: Optional[List[int]] = None,
        first_kernel_size: int = 7,
    ):
        """Base of ResNet with stacked residual blocks and a bottlnect residual block.
        Args:
        n_blocks (List[int]): number of blocks for each feature map size.
        n_channels (List[int]): the number of channels for each feature map size.
        bottlenecks (Optional[List[int]], optional): number of bottlenecks.
            Defaults to None. If none, residual blocsk are used.
        img_channels (int, optional): number of channels in the input. Defaults to 3.
        first_kernel_size (int, optional): kernel size of the initial convolution. Defaults to 7.
        """
        super().__init__()
        assert len(n_blocks) == len(n_channels)

        # if bottlenecks used, the number of channels in bottlenecks should be
        # provided for each feature map size.
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        # Initial convolution layer maps from img_channels to number of channels
        # in the first residual block (n_channels[0] )
        self.conv = nn.Conv2d(
            img_channels,
            n_channels[0],
            kernel_size=first_kernel_size,
            stride=2,
            padding=first_kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(n_channels[0])

        blocks = []

        # Number of channels from previous layer (or block)
        prev_channels = n_channels[0]

        # Loop through each feature map size
        for i, channels in enumerate(n_channels):
            # The first block for the new feature map size, will have a stride length of 2
            stride = 2 if len(blocks) == 0 else 1

            if bottlenecks is None:
                # residual blocks that maps from prev_channels to channels
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(
                    BottleneckResidualBlock(
                        prev_channels, bottlenecks[i], channels, stride=stride
                    )
                )

            # Change the number of channels
            prev_channels = channels

            # Add rest of the blocks - no change in feature map size or channels
            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResidualBlock(channels, channels, stride=1))
                else:
                    blocks.append(
                        BottleneckResidualBlock(
                            channels, bottlenecks[i], channels, stride=1
                        )
                    )
        # Stack the blocks
        self.blocks = nn.Sequential(*blocks)

        self.final_linear = nn.Linear(
            512, out_features=out_channels
        )  # check rigth input size here
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # x has shape [batch_size, img_channels, height, width]
        x = self.bn(self.conv(x))
        x = self.blocks(x)

        # Change x from shape [batch_size, channels, h, w] to [batch_size, channels, h * w]
        x = x.view(x.shape[0], x.shape[1], -1)

        # global average pooling across last dim (h*w)
        # x[0, 0, :].mean() and x.mean(dim=-1)[0, 0] will be the same
        x = x.mean(dim=-1)

        return self.softmax(self.final_linear(x))
