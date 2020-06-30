from arguments import arg_parser
from trainer import Trainer
from dataloader import Data_Loader
from torch.backends import cudnn
from utils import make_folder


def main(config):
    # For fast training
    cudnn.benchmark = True

    # Data loader
    data_loader = Data_Loader(config.train, config.dataset, config.image_path, config.imsize, config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    trainer = Trainer(data_loader.loader(), config)
    trainer.train()


if __name__ == '__main__':
    config = arg_parser()
    print(config)
    main(config)
