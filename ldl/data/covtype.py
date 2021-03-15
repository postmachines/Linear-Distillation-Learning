import pandas as pd
import numpy as np

import torch
import torchvision
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose
from sklearn.preprocessing import OneHotEncoder
from functools import partial

from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.datasets import fetch_covtype

def get_dataset(path_to_csv='data/Covtype/covtype.csv'):
    data = fetch_covtype(path_to_csv, as_frame=True)
    df = data['data'] 
    df=(df-df.mean())/df.std()
    df = pd.concat([df, data['target']], axis=1).rename(columns={'Cover_Type': 'target'})
    df['target'] = df['target'] - 1
    return df


class CustomDatasetFromCSV(torch.utils.data.Dataset):
    def __init__(self, path_to_csv='data/Covtype/covtype.csv'):
        self.data = get_dataset(path_to_csv=path_to_csv)
        self.X = self.data.drop(columns=['target'])
        self.y = self.data['target']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [torch.from_numpy(self.X.iloc[idx].values).float(), self.y[idx]]

    
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


def get_split(dataset, split='train'):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(2021))
    return [trainset, testset][split=='test']    
            
def load_class_rows(data, n_class):
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
    dataset = CustomDatasetFromCSV()
    customer_loader = torch.utils.data.DataLoader(get_split(dataset, split),
        batch_size=[464809, 116203][split =='test'], shuffle=False)

    n_classes = len(dataset.y.unique())
    
    data = {}
    for i_class in range(n_classes):
        data[i_class] = []
        
    for (x, y) in customer_loader:
        for i_class in range(n_classes):
            ny = y.numpy()
            data[i_class] = x[np.where(ny==i_class)]
        
    transforms = [partial(load_class_rows, data), partial(extract_episode, train_shot, test_shot)]
    transforms = compose(transforms)

    ds = TransformDataset(ListDataset([i for i in range(n_classes)]), transforms)
    sampler = EpisodicBatchSampler(len(ds), way, 1)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler)
    return loader


def get_data_loader(split='train', batch_size=32):
    # Load data
    dataset = CustomDatasetFromCSV()
    loader = torch.utils.data.DataLoader(get_split(dataset, split),
        batch_size=batch_size, shuffle=False)

    return loader
