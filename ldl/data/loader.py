from .mnist import get_episodic_loader as mnist_episodic_loader
from .omniglot import get_episodic_loader as omniglot_episodic_loader
from .cifar10 import get_episodic_loader as cifar10_episodic_loader

from .cifar10 import get_data_loader as cifar10_data_loader


def get_episodic_loader(dataset, way, train_shot, test_shot, x_dim=28, split='train', **kwargs):
    if dataset == "mnist":
        return mnist_episodic_loader(way, train_shot, test_shot, split)
    elif dataset == "omniglot":
        return omniglot_episodic_loader(way, train_shot, test_shot, x_dim, split, **kwargs)
    elif dataset == "cifar10":
        return cifar10_episodic_loader(way, train_shot, test_shot, x_dim, **kwargs)
    else:
        raise ValueError("Unknown dataset.")


def get_data_loader(dataset, split='train', batch_size=32):
    if dataset == "mnist":
        pass
    elif dataset == "omniglot":
        pass
    elif dataset == "cifar10":
        return cifar10_data_loader(split, batch_size)
    else:
        raise ValueError("Unknown dataset.")