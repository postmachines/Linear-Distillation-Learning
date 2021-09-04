from functools import partial
import numpy as np

import torch
import torchvision
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


def load_class_images(data, n_class):
    return data[n_class]


def extract_episode(n_support, n_query, data):
    """
    Extract data in an episodic format.
    Args:
        n_support (int): number of samples for support
        n_query (int): number of samples for query
        data (np.ndarray): numpy ndarray of data of single class

    Returns:

    """
    n_examples = data.size(0)

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = data[support_inds]
    xq = data[query_inds]

    return {
        'xs': xs,
        'xq': xq
    }


def get_episodic_loader(way, train_shot, test_shot, split, **kwargs):
    import torchvision.transforms as transforms

    transform = compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    mnist_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('../../data/CIFAR10/',
                                 train=split=='train',
                                 download=True,
                                 transform=transform),
        batch_size=[60000, 10000][split =='test'], shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data = {}
    for (x, y) in mnist_loader:
        for i_class in range(10):
            data[classes[i_class]] = x[y==i_class, :, :, :]

    transforms = [partial(load_class_images, data), partial(extract_episode, train_shot, test_shot)]
    transforms = compose(transforms)

    ds = TransformDataset(ListDataset(classes), transforms)
    sampler = EpisodicBatchSampler(len(ds), way, 1)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler)
    return loader


def get_data_loader(split='train', batch_size=32):
    import torchvision.transforms as transforms

    # Load data
    transform = compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5))])
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('../../data/CIFAR10/',
                                 train=split=='train',
                                 download=True,
                                 transform=transform),
        batch_size=batch_size, shuffle=True)

    return loader