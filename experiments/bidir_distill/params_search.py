import os
from time import time
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from train import run_experiment_full_test, run_experiment
from train_omniglot import run_experiment as run_experiment_omniglot
from train_table_dataset import run_experiment as run_experiment_table_ds


if __name__ == "__main__":
    print("GPU available: ", torch.cuda.is_available())

    configs = {
        'dataset': ['mnist'],
        'way': [10],
        'train_shot': [10, 50, 100, 200, 300],
        'test_shot': [1],
        'loss': [nn.MSELoss(reduction='none')],
        'epochs': [10],
        'trials': [50],
        'silent': [True],
        'split': ['test'],
        'x_dim': [28],
        'z_dim': [784, 2000],
        'lr_predictor': [1e-3, 1e-4, 5e-5],
        'lr_target': [1e-3, 1e-4, 5e-5],
        'channels': [1],
        'test_batch': [1],
        'save_data': [False],
        'in_alphabet': [False],
        'add_rotations': [False],
        'augmentation': [False],
        'gpu': [2],
        'table_dataset': [False],
        'full_test': [False]

    }
#     configs = {
#         'dataset': ['fashion_mnist'],
#         'way': [10],
#         'train_shot': [200],
#         'test_shot': [1],
#         'loss': [nn.MSELoss(reduction='none')],
#         'epochs': [10],
#         'trials': [50],
#         'silent': [True],
#         'split': ['test'],
#         'x_dim': [28],
#         'z_dim': [784, 2000],
#         'lr_predictor': [1e-3, 1e-4, 5e-5],
#         'lr_target': [1e-3, 1e-4, 5e-5],
#         'channels': [1],
#         'test_batch': [1],
#         'save_data': [False],
#         'in_alphabet': [False],
#         'add_rotations': [True],
#         'augmentation': [True],
#         'gpu': [2],
#         'full_test': [False],
#         'table_dataset': [False],
#     }
#     configs = {
#         'dataset': ['svhn'],
#         'way': [10],
#         'train_shot': [1, 10, 50, 100, 300],
#         'test_shot': [1],
#         'loss': [nn.MSELoss(reduction='none')],
#         'epochs': [10],
#         'trials': [100],
#         'silent': [True],
#         'split': ['test'],
#         'x_dim': [32],
#         'z_dim': [1024, 2000],
#         'lr_predictor': [1e-3, 1e-4, 5e-5],
#         'lr_target': [1e-3, 1e-4, 5e-5],
#         'channels': [3],
#         'test_batch': [1],
#         'save_data': [False],
#         'in_alphabet': [False],
#         'add_rotations': [True],
#         'augmentation': [True],
#         'full_test': [False],
#         'gpu': [1]
#     }

#     configs = {
#         'dataset': ['customer'],
#         'train_shot': [50, 200],
#         'x_dim': [41],
#         'z_dim': [1000],
#         'way': [2],
#         'test_shot': [1],
#         'loss': [nn.MSELoss(reduction='none')],
#         'epochs': [10],
#         'trials': [100],
#         'silent': [True],
#         'split': ['train'],
#         'lr_predictor': [1e-3, 1e-4, 5e-5],
#         'lr_target': [1e-3, 1e-4, 5e-5],
#         'test_batch': [1],
#         'save_data': [False],
#         'full_test': [False],
#         'gpu': [2],
#         'table_dataset': [True]
#     }

#     configs = {
#         'dataset': ['covtype'],
#         'train_shot': [200],
#         'x_dim': [54],
#         'z_dim': [1000, 2000],
#         'way': [7],
#         'test_shot': [1],
#         'loss': [nn.MSELoss(reduction='none')],
#         'epochs': [10],
#         'trials': [100],
#         'silent': [True],
#         'split': ['train'],
#         'lr_predictor': [1e-3, 1e-4, 5e-5],
#         'lr_target': [1e-3, 1e-4, 5e-5],
#         'test_batch': [1],
#         'save_data': [False],
#         'full_test': [False],
#         'gpu': [1],
#         'table_dataset': [True]
#     }
#     configs = {
#         'dataset': ['medical'],
#         'train_shot': [1, 10, 50, 100, 300],
#         'x_dim': [59],
#         'way': [6],
#         'test_shot': [1],
#         'loss': [nn.MSELoss(reduction='none')],
#         'epochs': [10],
#         'trials': [100],
#         'silent': [True],
#         'split': ['test'],
#         'lr_predictor': [1e-3, 1e-4, 5e-5],
#         'lr_target': [1e-3, 1e-4, 5e-5],
#         'test_batch': [1],
#         'save_data': [False],
#         'full_test': [False],
#         'gpu': [1],
#         'table_dataset': [True]
#     }


    # Create grid of parameters
    keys, values = zip(*configs.items())
    param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create resulting file if necessary
    ds_name = configs['dataset'][0]
    res_path = f"../../results/02-02-2021/results_{ds_name}_bidir.csv"
    if not os.path.exists(res_path):
        df = pd.DataFrame(columns=configs.keys())
        df.to_csv(res_path, index=False)
        
    table_dataset = configs['table_dataset'][0]
    if table_dataset:
        if ds_name in ['customer', 'covtype', 'medical']:
            exp_func = run_experiment_table_ds
        else:
            raise Exception("Unknown dataset!")
    else:
        if ds_name == 'omniglot':
            exp_func = run_experiment_omniglot
        elif ds_name in ['mnist', 'fashion_mnist', 'svhn']:
            if configs['full_test'][0]:
                exp_func = run_experiment_full_test
            else:
                exp_func = run_experiment
        else:
            raise Exception("Unknown dataset!")

    conf_durations = []
    for i, param in enumerate(param_grid):
#         if i < 15:
#             continue
        if len(conf_durations):
            time_estimate = (len(param_grid) - (i+1)) * np.mean(conf_durations) // 60
        else:
            time_estimate = '-'
        print(f"Configuration: ", param)
        print(f"Progress {i+1}/{len(param_grid)}. Estimated time until end: {time_estimate} min")
        time_start = time()
        mean_accuracy = exp_func(config=param)
        conf_durations.append(time() - time_start)
        df = pd.read_csv(res_path)
        df = df.append(pd.Series({**param, **{'accuracy': mean_accuracy,
                                              'duration_sec': conf_durations[-1]}}), ignore_index=True)
        df.to_csv(res_path, index=False)

