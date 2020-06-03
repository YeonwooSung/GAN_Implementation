import os

import torch
import torchvision

from tensorboardX import SummaryWriter


def export_image(G, save_path, global_step, resl, step, phase, img_num=5):
    latent_z = torch.randn(img_num, 512, 1, 1).cuda()
    img = G.forward(latent_z, "stabilization")

    for idx, img in enumerate(img):
        if not os.path.exists(save_path + "/fig"):
            os.mkdir(save_path + "/fig")
        torchvision.utils.save_image(img, "%s/fig/[%03d]_%s_%04d_%02d.png" % (save_path, resl, phase, global_step, idx))     # ~~~/out/fig/{epoch}_{idx}.png


class TensorboardLogger(SummaryWriter):
    def __init__(self, log_path):
        super().__init__(log_path)
        self.latent = torch.randn(4, 512, 1, 1)

    def log_hist(self, net, global_step):
        for n, p in net.named_parameters():
            n = n.replace(".", "/")
            n = "%s.%s" % (net._get_name(), n)
            self.add_histogram(n, p.detach().cpu().clone().numpy(), global_step)

    def log_scalar(self, tag, scalar, global_step):
        # single scalar
        if isinstance(scalar, (int, float)):
            self.add_scalar(tag, scalar, global_step)
        # scalar group
        elif isinstance(scalar, dict):
            self.add_scalars(tag, scalar, global_step)

    def log_image(self, G, mode, resl, global_step, img_num=16):
        img = G.forward(self.latent, mode)
        img = torchvision.utils.make_grid(img, nrow=int(img_num ** 0.5))
        self.add_image("%s/%s"%(resl, mode), img, global_step)
