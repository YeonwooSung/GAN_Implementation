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



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")

    return parser.parse_args()




def pullawayLoss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size, channels, latent_dim):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()

        # Encoder

        self.down = nn.Sequential(nn.Conv2d(channels, 64, 3, 2, 1), nn.ReLU())

        # calculate the down size and down dimension
        self.down_size = img_size // 2
        down_dim = 64 * (img_size // 2) ** 2

        # Encoder embedding
        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        embedding = self.embedding(out.view(out.size(0), -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out, embedding



if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)

    args = arg_parser()

    img_size, channels = args.img_size, args.channels
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    latent_dim = args.latent_dim
    sample_interval = args.sample_interval

    # define the image shape
    img_shape = (channels, img_size, img_size)

    # check if CUDA is available
    cuda = True if torch.cuda.is_available() else False

    # Reconstruction loss of AE
    pixelwise_loss = nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator(img_size=img_size, channels=channels, latent_dim=latent_dim)
    discriminator = Discriminator(img_size=img_size, channels=channels)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        pixelwise_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

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

    lr = args.lr
    b1, b2 = args.b1, args.b2

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    #--------------------
    #  Training
    #--------------------

    # BEGAN hyper parameters
    lambda_pt = 0.1
    margin = max(1, batch_size / 64.0)

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            #-----------------
            #  Train Generator
            #-----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)
            recon_imgs, img_embeddings = discriminator(gen_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullawayLoss(img_embeddings)

            g_loss.backward()
            optimizer_G.step()

            #---------------------
            #  Train Discriminator
            #---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_recon, _ = discriminator(real_imgs)
            fake_recon, _ = discriminator(gen_imgs.detach())

            d_loss_real = pixelwise_loss(real_recon, real_imgs)
            d_loss_fake = pixelwise_loss(fake_recon, gen_imgs.detach())

            d_loss = d_loss_real
            if (margin - d_loss_fake.data).item() > 0:
                d_loss += margin - d_loss_fake

            d_loss.backward()
            optimizer_D.step()

            #--------------
            # Log Progress
            #--------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
