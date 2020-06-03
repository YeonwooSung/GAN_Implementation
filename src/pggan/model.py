import torch
import torch.nn as nn
import torch.nn.functional as F

from hiddenLayers import PixelWiseNorm, EqualizedLR, MiniBatchSTD
from preset import resl_to_ch



#--------------------------------------------------------------------
# Generator
#--------------------------------------------------------------------

class ToRGBLayer(nn.Module):
    def __init__(self, resl, rgb_channel):
        super().__init__()
        _, in_c  = resl_to_ch[resl]

        self.conv = nn.Sequential(
            EqualizedLR(nn.Conv2d(in_c, rgb_channel, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class ReslBlock(nn.Module):
    def __init__(self, resl):
        super().__init__()
        in_c, out_c  = resl_to_ch[resl]

        self.conv = nn.Sequential(
            EqualizedLR(nn.Conv2d(in_c, out_c, 3, 1, 1)),
            PixelWiseNorm(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualizedLR(nn.Conv2d(out_c, out_c, 3, 1, 1)),
            PixelWiseNorm(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        up   = F.interpolate(x, scale_factor=2)
        conv = self.conv(up)
        return conv


class Generator(nn.Module):
    def __init__(self, resl=4, rgb_channel=3):
        super().__init__()
        self.resl = resl
        self.rgb_channel = rgb_channel

        in_c, out_c = resl_to_ch[resl]
        self.resl_blocks = nn.Sequential(
            # Original Repo using "tf.nn.conv2d_transpose"
            EqualizedLR(nn.ConvTranspose2d(in_c, out_c, 4)),
            # nn.Upsample(size=(4, 4)),
            # EqualizedLR(nn.Conv2d(in_c, out_c, 1, 1, 0)),
            PixelWiseNorm(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            EqualizedLR(nn.Conv2d(out_c, out_c, 3, 1, 1)),
            PixelWiseNorm(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.rgb_l = None
        self.rgb_h = ToRGBLayer(self.resl, self.rgb_channel)
        self.alpha = 0

    def forward(self, x, phase):
        if phase == "transition":
            return self.transition_forward(x)
        elif phase == "stabilization":
            return self.stabilization_forward(x)

    def grow_network(self):
        self.resl *= 2
        self.resl_blocks = nn.Sequential(*self.resl_blocks, ReslBlock(self.resl))
        self.rgb_l = self.rgb_h
        self.rgb_h = ToRGBLayer(self.resl, self.rgb_channel)
        self.alpha = 0

    def transition_forward(self, x):
        x = self.resl_blocks[:-1](x)

        # experiment candidate : apply rgb_l first and succeeding interpolate
        # low resolution path
        x_up = F.interpolate(x, scale_factor=2)
        rgb_l = self.rgb_l(x_up)

        # high resolution path
        x = self.resl_blocks[-1](x)
        rgb_h = self.rgb_h(x)

        return self.alpha * (rgb_h - rgb_l) + rgb_l

    def stabilization_forward(self, x):
        x = self.resl_blocks(x)
        rgb_h = self.rgb_h(x)
        return rgb_h

    def update_alpha(self, delta):
        self.alpha += delta
        self.alpha = min(1, self.alpha)



#--------------------------------------------------------------------
# Discriminator
#--------------------------------------------------------------------

class NoiseLayer(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.magnitude = 0
        self.loss_type = loss_type

    def forward(self, x):
        if self.loss_type != "lsgan":
            return x
        return x + (self.magnitude * torch.randn_like(x))


class FromRGBLayer(nn.Module):
    def __init__(self, resl, rgb_channel):
        super().__init__()
        _, out_c = resl_to_ch[resl]
        self.conv = EqualizedLR(nn.Conv2d(rgb_channel, out_c, 1))

    def forward(self, x):
        return self.conv(x)


class _DownReslBlock(nn.Module):
    def __init__(self, resl, loss_type):
        super().__init__()
        out_c, in_c  = resl_to_ch[resl]

        self.conv = nn.Sequential(
            NoiseLayer(loss_type),
            EqualizedLR(nn.Conv2d(in_c, out_c, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            NoiseLayer(loss_type),
            EqualizedLR(nn.Conv2d(out_c, out_c, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Discriminator(nn.Module):
    def __init__(self, loss_type='lsgan', resl=4, rgb_channel=3, group_size=4):
        super().__init__()
        self.loss_type = loss_type
        self.resl = resl
        self.rgb_channel = rgb_channel

        in_c, out_c = resl_to_ch[resl]
        self.resl_blocks = nn.Sequential(
            MiniBatchSTD(),
            NoiseLayer(loss_type),
            EqualizedLR(nn.Conv2d(in_c + 1, out_c, 3, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            NoiseLayer(loss_type),
            EqualizedLR(nn.Conv2d(in_c, out_c, 4, 1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Flatten(),
            EqualizedLR(nn.Linear(out_c, 1)),
        )

        self.rgb_l = None
        self.rgb_h = FromRGBLayer(resl, self.rgb_channel)
        self.alpha = 0

    def forward(self, x, phase):
        if phase == "transition":
            return self.transition_forward(x)
        elif phase == "stabilization":
            return self.stabilization_forward(x)

    def grow_network(self):
        self.resl *= 2
        self.resl_blocks = nn.Sequential(_DownReslBlock(self.loss_type, self.resl), *self.resl_blocks)
        self.rgb_l = self.rgb_h
        self.rgb_h = FromRGBLayer(self.resl, self.rgb_channel)
        self.alpha = 0

    def transition_forward(self, x):
        # low resolution path
        x_down = F.avg_pool2d(x, kernel_size=2)
        rgb_l = self.rgb_l(x_down)

        # high resolution path
        rgb_h = self.rgb_h(x)
        x_high = self.resl_blocks[0](rgb_h)

        x = self.alpha * (x_high - rgb_l) + rgb_l
        x = self.resl_blocks[1:](x)
        return x.view(-1)

    def stabilization_forward(self, x):
        x = self.rgb_h(x)
        x = self.resl_blocks(x)
        return x.view(-1)

    def update_alpha(self, delta):
        self.alpha += delta
        self.alpha = min(1, self.alpha)

    def update_noise(self, magnitude):
        for m in self.resl_blocks.modules():
            if isinstance(m, NoiseLayer):
                m.magnitude = magnitude
