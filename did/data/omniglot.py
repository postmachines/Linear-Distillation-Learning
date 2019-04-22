import os
import sys
import glob
from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose


OMNIGLOT_DATA_DIR  = os.path.join('data/omniglot')
OMNIGLOT_CACHE = { }


def convert_dict(k, v):
    return {k:v}


class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data


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


def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key])
    return d


def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    return d


def rotate_image(key, rot, d):
    d[key] = d[key].rotate(rot)
    return d


def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d


def load_class_images(d):
    if d['class'] not in OMNIGLOT_CACHE:
        alphabet, character, rot = d['class'].split('/')
        image_dir = os.path.join(OMNIGLOT_DATA_DIR, 'data', alphabet, character)

        class_images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if len(class_images) == 0:
            raise Exception("No images found for omniglot class {} at {}. Did you run download_omniglot.sh first?".format(d['class'], image_dir))

        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             partial(rotate_image, 'data', float(rot[3:])),
                                             partial(scale_image, 'data', 28, 28),
                                             partial(convert_tensor, 'data')]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            OMNIGLOT_CACHE[d['class']] = sample['data']
            break

    return { 'class': d['class'], 'data': OMNIGLOT_CACHE[d['class']] }


def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }


def load_omniglot(opt, splits):
    split_dir = os.path.join(OMNIGLOT_DATA_DIR, 'splits', opt['data.split'])

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        transforms = [partial(convert_dict, 'class'),
                      load_class_images,
                      partial(extract_episode, n_support, n_query)]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = []
        with open(os.path.join(split_dir, "{:s}.txt".format(split)), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))
        ds = TransformDataset(ListDataset(class_names), transforms)


        sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret


def get_n_classes(way, train_shot, test_shot, split='train'):
    """
    Return data loader of single episode.

    Args:
        way (int): number of different classes in data
        train_shot (int): number of samples per class in train
        test_shot (int): number of samples per class in test
        split (str): name of the split in 'vinyals' environment

    Returns (torch.utils.data.DataLoader): torch data loader

    """
    transforms = [partial(convert_dict, 'class'),
                  load_class_images,
                  partial(extract_episode, train_shot, test_shot)]
    transforms = compose(transforms)

    spliting = 'vinyals'
    split_dir = os.path.join(OMNIGLOT_DATA_DIR, 'splits', spliting)
    class_names = []
    with open(os.path.join(split_dir, "{:s}.txt".format(split)), 'r') as f:
        for class_name in f.readlines():
            class_names.append(class_name.rstrip('\n'))
    ds = TransformDataset(ListDataset(class_names), transforms)

    sampler = EpisodicBatchSampler(len(ds), way, 1)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)
    return loader