from typing import List

import torch
import torch.nn.functional as f
from torch import nn


def normalization(channels: int):
    # This is a helper function, with fixed number of groups and eps.
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)


def swish(x: torch.Tensor):
    """AKA SiLU
    x⋅σ(x)
    """
    return x * torch.sigmoid(x)


class GaussianDistribution:
    """parameters are the means and log of variances of the embedding of shape
    [batch_size, z_channels * 2, z_height, z_height]
    """

    def __init__(self, parameters: torch.Tensor):
        # Split mean and log of variance
        # mu is the mean point of the distribution
        # log_var is the logarithm of the variance of each dimension
        self.mu, self.log_var = torch.chunk(parameters, 2, dim=1)

        # Calculate standard deviation
        self.sigma = torch.exp(self.log_var / 2)

        # random input from a N(0,1) distribution
        self.epsilon = torch.randn_like(self.sigma)

    # Sample z from the distribution
    def sample(self):
        # reparamerization trick
        z = self.mu + (self.sigma * self.epsilon)

        return z


class ResnetBlock(nn.Module):
    # in_channels is the number of channels in the input
    # out_channels is the number of channels in the output
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # First normalization and convolution layer
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        # Second normalization and convolution layer
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        # in_channels to out_channels mapping layer for residual connection
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        # x is the input feature map with shape [batch_size, channels, height, width]

        h = x

        # First normalization and convolution layer
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        # Second normalization and convolution layer
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        # Map and add residual
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    # channels is the number of channels
    def __init__(self, channels: int):

        super().__init__()
        # Group norm
        self.norm = normalization(channels)

        # Query, key and value mappings
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)

        # Final 1×1 convolution layer
        self.proj_out = nn.Conv2d(channels, channels, 1)

        # Attention scaling factor
        self.scale = channels**-0.5

    def forward(self, x: torch.Tensor):
        # x is the tensor of shape [batch_size, channels, height, width]

        # group-norm
        x_norm = self.norm(x)

        # Get query, key and vector embeddings
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Reshape to query, key and vector embeedings from
        # [batch_size, channels, height, width] to
        # [batch_size, channels, height * width]
        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Compute softmax of the attention -> brings between 0,1
        attn = torch.einsum("bci,bcj->bij", q, k) * self.scale
        attn = f.softmax(attn, dim=2)

        # multiply by the value vector V
        out = torch.einsum("bij,bcj->bci", attn, v)

        # Reshape back to [batch_size, channels, height, width]
        out = out.view(b, c, h, w)

        # Final 1×1 convolution layer
        out = self.proj_out(out)

        # Add residual connection
        return x + out


class UpSample(nn.Module):
    # channels is the number of channels
    def __init__(self, channels: int):
        super().__init__()

        # 3×3 convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        # x is the input feature map with shape [batch_size, channels, height, width]

        # Up-sample by a factor of 2
        x = f.interpolate(x, scale_factor=2.0, mode="nearest")

        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    # channels is the number of channels
    def __init__(self, channels: int):
        super().__init__()

        # 3×3 convolution with stride length of 2 to down-sample by a factor of 2
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        # x is the input feature map with shape [batch_size, channels, height, width]

        # Add padding: (0, 1, 0, 1): pad last two dimensions by 1
        # i.e. (batch, channels, h, w) -> (batch, channels, h+1, w+1)
        # syntax: (1,1) pad last dimension by 1 on each side
        # (1, 1, 1, 1) pad last and 2nd to last dim by 1 on each side
        # (0, 1, 0, 1) pad last and 2nd to last dim by 1 on one side
        # (1, 1, 1, 1, 1, 1, 1, 1) pad each dim by 1 on each side
        x = f.pad(x, (0, 1, 0, 1), mode="constant", value=0)

        # Apply convolution
        return self.conv(x)


class Encoder(nn.Module):
    """Args:
    channels: number of channels in the first conv.
    channel_multipliers: multiplicative factors for the number of channels
                                in the subsequent blocks
    n_resnet_blocks: number of resnet layers at each resolution
    in_channels: the number of channels in the image
    z_channels: the number of channels in the embedding space
    """

    def __init__(
        self,
        channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        n_resnet_blocks: int = 2,
        in_channels: int = 1,
        z_channels: int = 512,
    ):
        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        n_resolutions = len(channel_multipliers)

        # initial conv mapping image_channels to channels
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Number of channels in each top level block
        # First block (out of 4) of encoder have two resnet blocks where the
        # channel dim is not changed since we have [1] + channel_multipliers
        # and channel_multiplers starts with [1], meaning the in and out channels
        # of the ResNet will be 1 * channels and 1 * channels, i.e. 64,64
        channels_list = [m * channels for m in [1] + channel_multipliers]

        # List of top-level blocks
        self.down = nn.ModuleList()

        # Create top-level blocks
        for i in range(n_resolutions):
            # Each top level block consists of multiple ResNet Blocks and down-sampling
            resnet_blocks = nn.ModuleList()

            # Add ResNet Blocks
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]

            # Top-level block
            down = nn.Module()
            down.block = resnet_blocks  # add resnet_blocks to down Module

            # Down-sampling at the end of each top level block except the last
            if i != n_resolutions - 1:
                down.downsample = DownSample(channels)
            else:
                down.downsample = nn.Identity()
            self.down.append(down)

        # Final ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # Map to embedding space with a 3×3 convolution
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 2 * z_channels, 3, stride=1, padding=1)

    def forward(self, img: torch.Tensor):

        # Map to channels with the initial convolution (img_channels, channels)
        x = self.conv_in(img)

        # Top-level blocks
        for down in self.down:
            # ResNet Blocks
            for block in down.block:
                x = block(x)

            # Down-sampling
            x = down.downsample(x)

        # Final ResNet blocks with attention
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        # Normalize and map to embedding space
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    """Args:
    channels: the number of channels in the final convolution layer
    channel_multipliers: the multiplicative factors for the number of channels
                        in the previous blocks, in reverse order
    n_resnet_blocks: the number of resnet layers at each resolution
    out_channels: the number of channels in the image
    z_channels: the number of channels in the embedding space
    """

    def __init__(
        self,
        channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        n_resnet_blocks: int = 2,
        out_channels: int = 1,
        z_channels: int = 512,
    ):

        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        num_resolutions = len(channel_multipliers)

        # Number of channels in each top level block, in the reverse order
        channels_list = [m * channels for m in channel_multipliers]

        # Number of channels in the top-level block
        channels = channels_list[-1]

        # initial 3×3 convolution layer that maps the embedding space to channels
        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        # ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # List of top-level blocks
        self.up = nn.ModuleList()

        # Create top-level blocks
        for i in reversed(range(num_resolutions)):
            # Each top level block consists of multiple ResNet Blocks and up-sampling
            resnet_blocks = nn.ModuleList()

            # Add ResNet Blocks
            for _ in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]

            # Top-level block
            up = nn.Module()
            up.block = resnet_blocks

            # Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upsample = UpSample(channels)
            else:
                up.upsample = nn.Identity()

            # Prepend to be consistent with the checkpoint
            self.up.insert(0, up)

        # Map to image space with a 3×3 convolution
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor):
        # z is the embedding tensor with shape
        # [batch_size, z_channels, z_height, z_height]

        # Map to channels with the initial convolution
        h = self.conv_in(z)

        # ResNet blocks with attention
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Top-level blocks
        for up in reversed(self.up):
            # ResNet Blocks
            for block in up.block:
                h = block(h)

            # Up-sampling
            h = up.upsample(h)

        # Normalize and map to image space
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(h)

        return img


class Autoencoder(nn.Module):
    """
    in_channels (int): channels in input image.
    z_channels (latent_channels) (int) : number of channels in the latent embedding space.
    out_channels (int): output channels
    channels: number of channels in first conv in encoder and last conv in decoder.
    channel_multipliers (list): channel dimension multiplier at the different levels.
    n_resnet_blocks (int): number of resnet blocks used in encoder and decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        z_channels: int = 512,
        out_channels: int = 1,
        channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        n_resnet_blocks: int = 2,
    ):
        super().__init__()
        self.encoder = Encoder(
            channels=channels,
            channel_multipliers=channel_multipliers,
            n_resnet_blocks=n_resnet_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
        )

        # Conv to map from embedding space to quantized emb. space moments (mean & var)
        self.quant_conv = nn.Conv2d(z_channels * 2, z_channels * 2, 1)

        self.decoder = Decoder(
            channels=channels,
            channel_multipliers=channel_multipliers,
            n_resnet_blocks=n_resnet_blocks,
            out_channels=out_channels,
            z_channels=z_channels,
        )

        # Conv to map from quantized emb. space back to embedding space
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

    def encode(self, img: torch.Tensor) -> GaussianDistribution:
        # Get embeddings with shape [batch_size, z_channels * 2, z_height, z_height]
        z = self.encoder(img)

        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)

        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        # z is the latent representation with
        # shape [batch_size, emb_channels, z_height, z_height]

        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)

        # Decode the image of shape [batch_size, channels, height, width]
        return self.decoder(z)

    def forward(self, img: torch.Tensor):
        # get a distribution
        z = self.encoder(img)  # 512
        moments = self.quant_conv(z)  # 1024

        # mean, std, 512 as moments is chunked
        self.distribution = GaussianDistribution(moments)

        # sample from it
        z = self.distribution.sample()

        z = self.post_quant_conv(z)

        img = self.decoder(z)

        # return mu and log_var as they are used in the KL divergence
        return img, self.distribution.mu, self.distribution.log_var
