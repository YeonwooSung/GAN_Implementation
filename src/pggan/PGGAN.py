import os
from glob import glob
import copy

import torch

from preset import resl_to_batch, resl_to_lr, resl_to_ch
from train_step import Train_LSGAN, Train_WGAN_GP


def get_optim(net, optim_type, resl, beta, decay, momentum, nesterov=True):
    lr = resl_to_lr[resl]
    return {
        "adam"    : torch.optim.Adam(net.parameters(), lr=lr, betas=beta, weight_decay=decay),
        "rmsprop" : torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=decay),
        "sgd"     : torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay, nesterov=True)
    }[optim_type]


class PGGAN:
    def __init__(self, arg, G, D, scalable_loader, torch_device, loss, tensorboard):
        self.arg = arg
        self.device = torch_device
        self.save_dir = arg.save_dir
        self.scalable_loader = scalable_loader

        self.img_num   = arg.img_num
        self.batch     = resl_to_batch[arg.start_resl]
        self.tran_step = self.img_num // self.batch
        self.stab_step = self.img_num // self.batch

        self.G = G

        self.G_ema = copy.deepcopy(G.module).cpu()
        self.G_ema.eval()
        for p in self.G_ema.parameters():
            p.requires_grad_(False)

        self.D = D
        self.optim_G = get_optim(self.G, self.arg.optim_G, self.arg.start_resl, self.arg.beta, self.arg.decay, self.arg.momentum)
        self.optim_D = get_optim(self.D, self.arg.optim_G, self.arg.start_resl, self.arg.beta, self.arg.decay, self.arg.momentum)

        self.tensorboard = tensorboard

        if loss == "lsgan":
            self.step = Train_LSGAN(self.G, self.D, self.optim_G, self.optim_D, self.arg.label_smoothing, self.batch, self.device)
        elif loss == "wgangp":
            self.step = Train_WGAN_GP(self.G, self.D, self.optim_G, self.optim_D, self.arg.gp_lambda, self.arg.eps_drift ,self.batch, self.device)

        self.load_resl = -1
        self.load_global_step = -1
        self.load()


    def save(self, global_step, resl, mode):
        """Save current step model
        Save Elements:
            model_type : arg.model
            start_step : current step
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score
        Parameters:
            step : current step
            filename : model save file name
        """
        torch.save({"global_step" : global_step,
                    "resl" : resl,
                    "G" : self.G.state_dict(),
                    "G_ema" : self.G_ema.state_dict(),
                    "D" : self.D.state_dict(),
                    "optim_G" : self.optim_G.state_dict(),
                    "optim_D" : self.optim_D.state_dict(),
                    }, self.save_dir + "/step_%07d_resl_%d_%s.pth.tar" % (global_step, resl, mode))
        print("Model saved %d step" % (global_step))

    def load(self, filename=None):
        """ Model load. same with save"""
        if filename is None:
            # load last epoch model
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[-1])

        file_path = self.save_dir + "/" + filename

        if os.path.exists(file_path) is True:
            print("Load %s to %s File" % (self.save_dir, filename))
            ckpoint = torch.load(file_path)

            self.load_resl = ckpoint["resl"]

            resl = self.arg.start_resl
            while resl < self.load_resl:
                self.G.module.grow_network()
                self.D.module.grow_network()
                self.G_ema.grow_network()
                self.G.to(self.device)
                self.D.to(self.device)
                resl *= 2

            self.G.load_state_dict(ckpoint["G"])
            self.G_ema.load_state_dict(ckpoint["G_ema"])
            self.D.load_state_dict(ckpoint["D"])
            self.optim_G.load_state_dict(ckpoint['optim_G'])
            self.optim_D.load_state_dict(ckpoint['optim_D'])
            self.load_global_step = ckpoint["global_step"]
            print("Load Model, Global step : %d / Resolution : %d " % (self.load_global_step, self.load_resl))

        else:
            print("Load Failed, not exists file")



    def grow_architecture(self, resl, global_step):
        resl *= 2

        self.batch     = resl_to_batch[resl]
        self.stab_step = (self.img_num // self.batch) * resl_to_ch[resl]
        self.tran_step = (self.img_num // self.batch) * resl_to_ch[resl]

        self.optim_G.param_groups = []
        self.optim_G.add_param_group({"params": list(self.G.parameters())})
        self.optim_D.param_groups = []
        self.optim_D.add_param_group({"params": list(self.D.parameters())})

        lr = resl_to_lr[resl]
        for x in self.optim_G.param_groups + self.optim_D.param_groups:
            x["lr"] = lr
        self.step.grow(self.batch, self.optim_G, self.optim_D)


        # When the saved model is loaded, self.load() already grows the architecture
        # To prevent additional growing, this condition is required
        if global_step >= self.load_global_step:
            self.G.module.grow_network()
            self.G_ema.grow_network()
            self.D.module.grow_network()
            self.G.to(self.device)
            self.D.to(self.device)
            torch.cuda.empty_cache()
            return resl
        else:
            self.G.module.alpha = 0
            self.G_ema.alpha = 0
            self.D.module.alpha = 0
            return resl


    def update_ema(self):
        with torch.no_grad():
            named_param = dict(self.G.module.named_parameters())
            for k, v in self.G_ema.named_parameters():
                param = named_param[k].detach().cpu()
                v.copy_(self.arg.ema_decay * v + (1 - self.arg.ema_decay) * param)


    def train(self):
        # Initialize Train
        global_step, resl = 0, self.arg.start_resl
        loader = self.scalable_loader(resl)

        def _step(step, loader, mode, LOG_PER_STEP=50):
            # When the saved model is loaded,
            # skips network train until loaded step
            nonlocal global_step
            if global_step <= self.load_global_step:
                global_step += 1
                return
                
            input_, _ = next(loader)
            input_ = input_.to(self.device)
            log_D = self.step.train_D(input_, mode, d_iter=self.arg.d_iter)
            log_G = self.step.train_G(mode)
            self.update_ema()

            # Save images and record logs
            if (step % LOG_PER_STEP) == 0:
                print("[% 6d/% 6d : % 3.2f %%]" % (step, self.tran_step, (step / self.tran_step) * 100))
                self.G_ema.eval()
                with torch.no_grad():
                    self.tensorboard.log_image(self.G_ema, mode, resl, global_step)
                self.tensorboard.log_scalar("Loss/%d" % (resl), {**log_D, **log_G}, global_step)

            if (step % (LOG_PER_STEP * 10)) == 0:
                self.save(global_step, resl, mode)
            global_step += 1


        # Stabilization on initial resolution (default: 4 * 4)
        for step in range(self.stab_step):
            _step(step, loader, "stabilization")

        while (resl < self.arg.end_resl):
            # Grow and update resolution, batch size, etc. Load the models on GPUs
            resl = self.grow_architecture(resl, global_step)
            loader = self.scalable_loader(resl)
            for step in range(self.tran_step):
                _step(step, loader, "transition")
                self.G.module.update_alpha(1 / self.tran_step)
                self.G_ema.update_alpha(1 / self.tran_step)
                self.D.module.update_alpha(1 / self.tran_step)

            # Stabilization
            for step in range(self.stab_step):
                _step(step, loader, "stabilization")

        for step in range(self.arg.extra_training_img_num):
            _step(step, loader, "stabilization")
