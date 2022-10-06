"""Unet model for training diffusion model
"""
from typing import List, Tuple, Union
import math
import torch
from einops import rearrange
from inspect import isfunction
from torch import nn, einsum


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
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
        self.conv2d = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResNetBlock(nn.Module):
    """https://miro.medium.com/max/828/1*D0F3UitQ2l5Q0Ak-tjEdJg.png
    Two conv blocks with group norms and SiLU activations + a residual connection.
    A time embedding is added in between the conv blocks."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(), nn.Linear(in_features=time_emb_dim, out_features=dim_out)
            )
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        # nn.Identity is a placeholder that is argument insensitive
        self.res_conn_conv = (
            nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h  # extend last two dimesions

        h = self.block2(h)
        return h + self.res_conn_conv(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

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
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downs = nn.ModuleList([])
        for i in range(len(dims) - 2):
            self.downs.append(
                nn.ModuleList(
                    [
                        ResNetBlock(dims[i], dims[i + 1], time_emb_dim=time_emb_dim),
                        Residual(PreNorm(dims[i + 1], LinearAttention(dims[i + 1]))),
                        self.pool,
                    ]
                )
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> List:
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

        self.ups = nn.ModuleList([])
        for i in range(len(dims) - 2):
            self.ups.append(
                nn.ModuleList(
                    [
                        ResNetBlock(dims[i], dims[i + 1], time_emb_dim=time_emb_dim),
                        Residual(PreNorm(dims[i + 1], LinearAttention(dims[i + 1]))),
                        nn.ConvTranspose2d(
                            dims[i], dims[i + 1], kernel_size=2, stride=2
                        ),
                    ]
                )
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
    by a linear layer, followed by an activation function,
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


class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4, 8),
        out_channels=1,
        with_time_emb: bool = True,
        num_classes: int = None,
    ) -> None:
        super().__init__()
        self.channels = [image_channels] + list(map(lambda x: x * n_channels, ch_mults))

        if with_time_emb:
            self.time_dim = n_channels * 4
            self.time_emb = TimeEmbedding(self.time_dim)  # why * 4
        else:
            self.time_dim = None
            self.time_emb = None

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.time_dim)

        self.encoder = Encoder(dims=self.channels, time_emb_dim=self.time_dim)

        self.bottleneck = ResNetBlock(
            dim=self.channels[-2], dim_out=self.channels[-1], time_emb_dim=self.time_dim
        )
        self.bottleneck_attn = Residual(
            PreNorm(self.channels[-1], Attention(self.channels[-1]))
        )

        self.decoder = Decoder(
            dims=list(reversed(self.channels)), time_emb_dim=self.time_dim
        )

        self.final_conv = nn.Sequential(
            ResNetBlock(dim=n_channels, dim_out=n_channels),
            nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        t = self.time_emb(t) if exists(self.time_emb) else None

        if y is not None:
            t += self.label_emb(y)

        # downsample
        x, enc_ftrs = self.encoder(x, t)

        # bottleneck
        x = self.bottleneck(x, t)
        x = self.bottleneck_attn(x)

        # upsample
        x = self.decoder(x, enc_ftrs, t)

        # final layer
        out = self.final_conv(x)

        return out
