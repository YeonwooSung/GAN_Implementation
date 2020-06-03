import random
from PIL import Image

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from preset import resl_to_batch


class ScalableLoader:
    def __init__(self, path, shuffle=True, drop_last=False, num_workers=4, shuffled_cycle=True):
        self.path = path
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.shuffled_cycle = shuffled_cycle

    def __call__(self, resl):
        batch = resl_to_batch[resl]

        transform = transforms.Compose([transforms.Resize(size=(resl, resl), interpolation=Image.NEAREST), transforms.ToTensor()])

        root = self.path + str(max(64, resl))
        print("Data root : %s" % root)

        loader = DataLoader(
            dataset=ImageFolder(root=root, transform=transform),
            batch_size=batch,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers
        )

        loader = self.cycle(loader)
        return loader

    def cycle(self, loader):
        while True:
            for element in loader:
                yield element
            if self.shuffled_cycle:
                random.shuffle(loader.dataset.imgs)
