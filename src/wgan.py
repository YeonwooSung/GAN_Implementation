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


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. (the clipping parameter c) weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    return parser.parse_args()


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

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
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# In fact, if you see the paper, you would find that they replace the original discriminator with critic.
# However, in this code, I will name the critic as "discriminator" to keep the consistency with other GANs.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity



if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)

    args = argument_parser()
    img_shape = (args.channels, args.img_size, args.img_size)
    latent_dim = args.latent_dim

    # check if CUDA is available
    CUDA = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator()

    if CUDA:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    learning_rate = args.lr
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate)

    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


    #---------------------------------------------------------------------------------------------------------
    # Training Generator and Discriminator
    #---------------------------------------------------------------------------------------------------------

    batches_done = 0
    n_epochs = args.n_epochs
    clipping_value = args.clip_value
    neg_clipping_value = -(clipping_value)
    n_critic = args.n_critic

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))


            #--------------------------
            # train discriminator
            #--------------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = - torch.mean(discriminator(real_imgs) + torch.mean(discriminator(fake_imgs)))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(neg_clipping_value, clipping_value)


            # Train generator for every n_critic iterations
            if i % n_critic != 0:
                continue
            

            #--------------------------
            # train generator
            #--------------------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = - torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1
