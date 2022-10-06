from torch import nn


def block(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())


class SparseAutoEncoder(nn.Module):
    def __init__(self, dims: list() = [784, 256, 128, 64, 32, 16]):
        # super call delegates function call to nn.Module(parent class
        super().__init__()
        blocks_enc = [
            block(in_dim, out_dim) for (in_dim, out_dim) in zip(dims[:-1], dims[1:])
        ]
        blocks_dec = [
            block(in_dim, out_dim)
            for (in_dim, out_dim) in zip(
                list(reversed(dims))[:-1], list(reversed(dims))[1:]
            )
        ]
        self.encoder = nn.Sequential(
            *blocks_enc
        )  # Sequential does not take a list so we decompose using *
        self.decoder = nn.Sequential(*blocks_dec)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)

        return x
