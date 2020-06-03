import torch
import torch.nn as nn
from torch.autograd import grad


class Train_LSGAN:
    def __init__(self, G, D, optim_G, optim_D, label_smoothing, batch, device):
        self.G = G
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D

        self.loss = nn.MSELoss()
        self.device = device

        self.batch = batch
        self.label_smoothing = label_smoothing
        self.ones  = torch.ones(batch).to(device)
        self.zeros = torch.zeros(batch).to(device)

        self.d_hat = 0
        self.last_d_hat = 0
        self.noise = 0

        # d_pres_hat = 0.1 * d_out + 0.9 * d_last_hat
        # noise = 0.2 * (max(0, d_pres_hat - 0.5) ** 2)

    def train_D(self, x, mode, d_iter=1):
        for _ in range(d_iter):
            latent_z = torch.randn(self.batch, 512, 1, 1).normal_().to(self.device)

            fake_x = self.G.forward(latent_z, mode)
            fake_y = self.D.forward(fake_x, mode)
            fake_loss = self.loss(fake_y, self.zeros)

            real_y = self.D.forward(x, mode)
            real_loss = self.loss(real_y, self.ones - self.label_smoothing)

            self.optim_D.zero_grad()
            loss_D = fake_loss + real_loss
            loss_D.backward()
            self.optim_D.step()
            
            self.last_d_hat = self.d_hat
            self.d_hat = 0.1 * real_y.mean().item() + 0.9 * self.last_d_hat
            noise = 0.2 * (max(0, self.d_hat - 0.5) ** 2)
            self.D.module.update_noise(noise)

        return {"loss_D"    : loss_D.item(),
                "fake_loss" : fake_loss.item(),
                "real_loss" : real_loss.item(),
                "noise_mag" : noise,
                "mean_y"    : real_y.mean(),
                "alpha_D"   : self.D.module.alpha}

    def train_G(self, mode):
        latent_z = torch.randn(self.batch, 512, 1, 1).normal_().to(self.device)

        fake_x = self.G.forward(latent_z, mode)
        fake_y = self.D.forward(fake_x, mode)

        self.optim_G.zero_grad()
        loss_G = self.loss(fake_y, self.ones)
        loss_G.backward()
        self.optim_G.step()

        return {"loss_G"    : loss_G.item(),
                "alpha_G"   : self.G.module.alpha}

    def grow(self, batch, optim_G, optim_D):
        self.batch   = batch
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.ones  = torch.ones(batch).to(self.device)
        self.zeros = torch.zeros(batch).to(self.device)


class Train_WGAN_GP:
    def __init__(self, G, D, optim_G, optim_D, gp_lambda, eps_drift, batch, device):
        self.G = G
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D

        self.device = device

        self.batch = batch
        self.gp_lambda = gp_lambda
        self.eps_drift = eps_drift

    def get_gp(self, x, fake_x, mode):
        alpha = torch.rand(self.batch, 1, 1, 1).to(self.device)

        x_hat = alpha * x.detach() + (1 - alpha) * fake_x.detach()
        x_hat.requires_grad_(True)

        pred_hat = self.D(x_hat, mode)
        gradients = grad(outputs=pred_hat, inputs=x_hat,
                         grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        grad_norm = gradients.view(self.batch, -1).norm(2, dim=1)
        return self.gp_lambda * grad_norm.sub(1).pow(2).mean()

    def train_D(self, x, mode, d_iter):
        for _ in range(d_iter):
            latent_z = torch.randn(self.batch, 512, 1, 1).normal_().to(self.device)

            fake_x = self.G(latent_z, mode)
            fake_y = self.D(fake_x, mode)
            fake_loss = fake_y.mean()

            real_y = self.D(x, mode)
            real_loss = real_y.mean()

            drift = real_y.pow(2).mean() * self.eps_drift

            gp = self.get_gp(x, fake_x, mode)

            self.optim_D.zero_grad()
            loss_D = fake_loss - real_loss + drift + gp
            loss_D.backward()
            self.optim_D.step()

        return {"loss_D"    : loss_D,
                "fake_loss" : fake_loss,
                "real_loss" : real_loss,
                "drift"     : drift,
                "gp"        : gp,
                "alpha_D"   : self.D.module.alpha}

    def train_G(self, mode):
        latent_z = torch.randn(self.batch, 512, 1, 1).normal_().to(self.device)
        fake_x = self.G(latent_z, mode)
        fake_y = self.D(fake_x, mode)

        self.optim_G.zero_grad()
        loss_G = -1 * fake_y.mean()
        loss_G.backward()
        self.optim_G.step()
        return {"loss_G"    : loss_G,
                "alpha_G"   : self.G.module.alpha}

    def grow(self, batch, optim_G, optim_D):
        self.batch   = batch
        self.optim_D = optim_D
        self.optim_G = optim_G
