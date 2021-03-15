from .mnist import get_episodic_loader as mnist_episodic_loader
from .fashion_mnist import get_episodic_loader as fashion_mnist_episodic_loader
from .omniglot import get_episodic_loader as omniglot_episodic_loader
from .svhn import get_episodic_loader as svhn_episodic_loader
from .cifar10 import get_episodic_loader as cifar10_episodic_loader
from .cifar10 import get_data_loader as cifar10_data_loader
from .customer import get_episodic_loader as customer_episodic_loader
from .covtype import get_episodic_loader as covtype_episodic_loader

def get_episodic_loader(dataset, way, train_shot, test_shot, x_dim=28, split='train', **kwargs):
    if dataset == "mnist":
        return mnist_episodic_loader(way, train_shot, test_shot, split)
    elif dataset == "fashion_mnist":
        return fashion_mnist_episodic_loader(way, train_shot, test_shot, split)
    elif dataset == "svhn":
        return svhn_episodic_loader(way, train_shot, test_shot, split)    
    elif dataset == "omniglot":
        return omniglot_episodic_loader(way, train_shot, test_shot, x_dim, split, **kwargs)
    elif dataset == "cifar10":
        return cifar10_episodic_loader(way, train_shot, test_shot, x_dim, **kwargs)
    elif dataset == "customer":
        return customer_episodic_loader(way, train_shot, test_shot, split)
    elif dataset == 'covtype':
        return covtype_episodic_loader(way, train_shot, test_shot, split)
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