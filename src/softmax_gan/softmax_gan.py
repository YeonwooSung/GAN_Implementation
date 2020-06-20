import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    return parser.parse_args()


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()

        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img_shape = self.img_shape
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_size ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)

        return validity


def log(x):
    return torch.log(x + 1e-8)


if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)

    args = arg_parse()

    img_shape = (args.channels, args.img_size, args.img_size)
    CUDA = True if torch.cuda.is_available() else False

    # get arguments
    img_size = args.img_size
    latent_dim = args.latent_dim
    lr = args.lr
    b1, b2 = args.b1, args.b2
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    sample_interval = args.sample_interval


    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(img_shape, latent_dim)
    discriminator = Discriminator(img_size)

    if CUDA:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    

    # ----------
    #  Training
    # ----------

    print('Start training the Softmax GAN')

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            g_target = 1 / (batch_size * 2)
            d_target = 1 / batch_size

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            # Generate a batch of images
            gen_imgs = generator(z)

            d_real = discriminator(real_imgs)
            d_fake = discriminator(gen_imgs)

            # Partition function
            Z = torch.sum(torch.exp(-d_real)) + torch.sum(torch.exp(-d_fake))

            # Calculate loss of discriminator and update
            d_loss = d_target * torch.sum(d_real) + log(Z)
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            # Calculate loss of generator and update
            g_loss = g_target * (torch.sum(d_real) + torch.sum(d_fake)) + log(Z)
            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
