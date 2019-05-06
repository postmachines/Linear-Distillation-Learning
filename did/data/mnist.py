import torch
import torchvision


def get_data_loader(split='train', batch_size=32):
    # Load data
    loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/MNIST/', train=split=='train',
                               download=True,
                               transform=torchvision.transforms.ToTensor(),
                       ),
        batch_size=batch_size, shuffle=True)

    return loader


