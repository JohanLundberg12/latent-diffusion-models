"""Unet model for training diffusion model
"""
from typing import List, Tuple, Union
import math
import torch
from einops import rearrange
from torch import nn, einsum


def exists(x):
    return x is not None


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        """Creates sinusoidal time step positional embeddings
        Args:
            dim (int): the channel dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device

        # half the channel is sin and the other is cos
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # [None, :] changes shape to [1, dim], [:, None] to [dim, 1]
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    """A conv2d block with group norm followed by sigmoid linear unit, elment wise."""

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.activation = nn.SiLU()
        self.conv2d = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)

    # it is conv(act(norm(x))) because of the in_conv in UNet
    def forward(self, x):
        return self.conv2d(self.activation(self.norm(x)))


class ResNetBlock(nn.Module):
    """https://miro.medium.com/max/828/1*D0F3UitQ2l5Q0Ak-tjEdJg.png
    Two conv blocks with group norms and SiLU activations + a residual connection.
    A time embedding is added in between the conv blocks."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        # time step embedding
        self.mlp_t = (
            nn.Sequential(
                nn.SiLU(), nn.Linear(in_features=time_emb_dim, out_features=dim_out)
            )
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        # nn.Identity is a placeholder that is argument insensitive
        self.shortcut = (
            nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None):
        h = self.block1(x)

        if exists(self.mlp_t) and exists(time_emb):
            # create time step embeddings
            time_emb = self.mlp_t(time_emb)

            # add time step embeddings but first extend last two dimensions
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        # final conv layer with norm and activation
        h = self.block2(h)

        # add residual connection
        return h + self.shortcut(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)  # (1, dim) equivalent to layer norm

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Encoder(nn.Module):
    """Down part of unet but doubleconv replaced with Resnet block
    followed by a residual connection f(x) + x where f is linear attention.
    Finally a maxpool2d."""

    def __init__(self, dims: List[int], time_emb_dim: int = None) -> None:
        """
        Args:
            channels (int): number of in channels in the ResNet feature map.
                For instance: 1 or 3.
            dims (List[int]): the number of channels at the next levels of the UNet.
                For instance: [64, 128, 256, 512]
            time_emb_dim (int, optional): time step embedding. Defaults to None.
        """
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        levels = len(dims)

        self.downs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResNetBlock(dims[i], dims[i + 1], time_emb_dim=time_emb_dim),
                        Residual(PreNorm(dims[i + 1], LinearAttention(dims[i + 1]))),
                        self.pool,
                    ]
                )
                for i in range(levels - 1)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> List:
        skip_connections = list()

        for block1, attn, downsample in self.downs:
            x = block1(x, t)
            x = attn(x)
            skip_connections.append(x)
            x = downsample(x)

        return x, skip_connections


class Decoder(nn.Module):
    """Decoder part of unet, similar to encoder but with Convtranspose."""

    def __init__(self, dims: List, time_emb_dim=None) -> None:
        super().__init__()

        levels = len(dims)

        self.ups = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        # dims[i] + dims[i + 1] because we add the skip connection
                        ResNetBlock(
                            dims[i] + dims[i + 1],
                            dims[i + 1],
                            time_emb_dim=time_emb_dim,
                        ),
                        Residual(PreNorm(dims[i + 1], LinearAttention(dims[i + 1]))),
                        nn.ConvTranspose2d(
                            dims[i], dims[i + 1], kernel_size=2, stride=2
                        ),
                    ]
                )
                for i in range(levels - 1)
            ]
        )

    def forward(
        self, x: torch.Tensor, skip_connections: List, t: torch.Tensor
    ) -> torch.Tensor:
        for block1, attn, upsample in self.ups:
            x = upsample(x)
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block1(x, t)
            x = attn(x)
        return x


class TimeEmbedding(nn.Module):
    """
    Embedding for time step $t$
    Combined of a sinusoidalpositional embedding followed
    by a linear layer, followed by a GELU activation function,
    (could also be SiLU),
    followed by a linear layer.
    """

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim=self.n_channels // 4),
            nn.Linear(in_features=self.n_channels // 4, out_features=self.n_channels),
            nn.GELU(),
            nn.Linear(in_features=self.n_channels, out_features=self.n_channels),
        )

    def forward(self, t: torch.Tensor):
        emb = self.time_mlp(t)

        return emb


class BottleNeck(nn.Module):
    def __init__(self, channels: int, time_channels: int) -> None:
        super().__init__()
        self.res1 = ResNetBlock(
            dim=channels, dim_out=channels, time_emb_dim=time_channels
        )
        self.attn = Residual(PreNorm(channels, Attention(channels)))
        self.res2 = ResNetBlock(
            dim=channels, dim_out=channels, time_emb_dim=time_channels
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res2(self.attn(self.res1(x)))

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int = 64,
        channel_multipliers: Union[Tuple[int, ...], List[int]] = (1, 2, 4, 8),
        with_time_emb: bool = True,
        num_classes: int = None,
    ) -> None:
        """
        Args:
        in_channels (int): number of channels in the input feature map
        out_channels (int): number of channels in the output feature map
        channels (int, optional): base channel count for the model.
            Defaults to 64.
        channel_multipliers (Union[Tuple[int, ...], List[int]], optional):
            Defaults to (1, 2, 4, 8).
        with_time_emb (bool, optional): Whether to use time embeddings or not.
            Defaults to True.
        num_classes (int, optional): number of classes.
            Defaults to None.
        """
        super().__init__()
        self.channels_list = [channels] + [channels * m for m in channel_multipliers]
        self.num_classes = num_classes

        if with_time_emb:
            d_time_emb = channels * 4  # why * 4?
            self.time_emb = TimeEmbedding(d_time_emb)
        else:
            d_time_emb = None
            self.time_emb = None

        # if conditioning on classes, create embedding to add to the time embedding
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, d_time_emb)

        self.initial_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        self.encoder = Encoder(dims=self.channels_list, time_emb_dim=d_time_emb)

        self.bottleneck = BottleNeck(
            channels=self.channels_list[-1],
            time_channels=d_time_emb,
        )

        self.decoder = Decoder(
            dims=list(reversed(self.channels_list)),
            time_emb_dim=d_time_emb,
        )

        self.final_conv = nn.Sequential(
            ResNetBlock(dim=channels, dim_out=channels),
            nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=1),
        )

    def encode(self, x: torch.Tensor, t: torch.Tensor = None):
        x, enc_frts = self.encoder(x, t)
        h = self.bottleneck(x, t)

        return h, enc_frts

    def decode(self, x: torch.Tensor, enc_ftrs: list(), t: torch.Tensor):
        x = self.decoder(x, enc_ftrs, t)

        return x

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
        x_noisy (torch.Tensor): input tensor of shape [batch_size, channels, width, height]
        t (torch.Tensor): timesteps  of shape [batch_size]
        y (torch.Tensor): classes.
        """
        t_emb = self.time_emb(t) if exists(self.time_emb) else None

        if y is not None:
            t_emb += self.label_emb(y)

        x = self.initial_conv(x_noisy)

        # downsample
        z, enc_ftrs = self.encode(x, t_emb)

        # upsample
        x = self.decode(z, enc_ftrs, t_emb)

        # final layer
        out = self.final_conv(x)

        return out
