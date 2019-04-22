import torch
import torchvision


def load(opt, splits=None):
    # Load data
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/MNIST/', train=True, download=True,
                       transform=torchvision.transforms.ToTensor(),
                       ),
        batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/MNIST/', train=False, download=True,
                        transform=torchvision.transforms.ToTensor(),
                       ),
    batch_size=1, shuffle=True)

    ret = {
        'train': train_loader,
        'test': test_loader
    }
    return ret
