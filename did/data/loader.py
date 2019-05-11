from .mnist import get_episodic_loader as mnist_episodic_loader
from .omniglot import get_episodic_loader as omniglot_episodic_loader


def get_episodic_loader(dataset, way, train_shot, test_shot, x_dim=28, split='train', **kwargs):
    if dataset == "mnist":
        return mnist_episodic_loader(way, train_shot, test_shot, split)
    elif dataset == "omniglot":
        return omniglot_episodic_loader(way, train_shot, test_shot, x_dim, split, **kwargs)
    else:
        raise ValueError("Unknown dataset.")