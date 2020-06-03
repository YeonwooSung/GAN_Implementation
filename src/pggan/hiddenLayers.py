import torch
import torch.nn as nn



class EqualizedLR(nn.Module):
    def __init__(self, layer):
        super().__init__()

        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        layer.bias.data.fill_(0)

        self.wscale = layer.weight.data.detach().pow(2.).mean().sqrt()
        layer.weight.data /= self.wscale

        self.layer = layer

    def forward(self, x):
        return self.layer(x * self.wscale)


class MiniBatchSTD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        std = torch.std(x).expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, std], dim=1)


class PixelWiseNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_square_mean = x.pow(2).mean(dim=1, keepdim=True)
        denom = torch.rsqrt(x_square_mean + 1e-8)
        return x * denom
