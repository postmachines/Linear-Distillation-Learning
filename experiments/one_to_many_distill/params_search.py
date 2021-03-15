import os
from time import time
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from train import run_experiment
from train_full_test import run_experiment_full_test
from train_table_customer import run_experiment as run_experiment_customer
from train_table_covtype import run_experiment as run_experiment_covtype

if __name__ == "__main__":
    print("GPU available: ", torch.cuda.is_available())

#     configs = {
#         'dataset': ['mnist'],
#         'epochs': [3, 10],
#         'way': [10],
#         'train_shot': [1, 5, 10, 50, 100, 200, 300],
#         'test_shot': [1],
#         'x_dim': [28], # ATTENTION: Due to the cached nature of dataloader this parameter should be set in signle value per run
#         'z_dim': [784, 2000],
#         'dld': [True],
#         'optimizer': ['adam'],
#         'lr': [1e-3, 1e-4, 5e-5],
#         'initialization': ['xavier_normal'],
#         'channels': [1],
#         'loss': [nn.MSELoss(reduction='none')],
#         'trials': [100],
#         'silent': [True],
#         'split': ['test'],
#         'in_alphabet': [False],
#         'add_rotations': [True],
#         'gpu': [1],
#         'test_batch': [2000],
#         'full_test': [False],
#         'save_data': [False]
#     }
#     configs = {
#         'dataset': ['svhn'],
#         'epochs': [3, 10],
#         'way': [10],
#         'train_shot': [1, 5, 10, 100,  300],
#         'test_shot': [1],
#         'x_dim': [32], # ATTENTION: Due to the cached nature of dataloader this parameter should be set in signle value per run
#         'z_dim': [1024, 2000],
#         'dld': [True],
#         'optimizer': ['adam'],
#         'lr': [1e-3, 1e-4, 5e-5],
#         'initialization': ['xavier_normal'],
#         'channels': [3],
#         'loss': [nn.MSELoss(reduction='none')],
#         'trials': [100],
#         'silent': [False],
#         'split': ['test'],
#         'in_alphabet': [False],
#         'add_rotations': [True],
#         'gpu': [1],
#         'test_batch': [2000],
#         'full_test': [False],
#         'save_data': [False]
#     }

#     configs = {
#         'dataset': ['customer'],
#         'epochs': [3, 10],
#         'train_shot': [1, 5, 10, 100, 300],
#         'test_shot': [2],
#         'dld': [True],
#         'optimizer': ['adam'],
#         'lr': [1e-2, 1e-3, 1e-4, 5e-5],
#         'initialization': ['xavier_normal'],
#         'loss': [nn.MSELoss(reduction='none')],
#         'trials': [100],
#         'silent': [False],
#         'split': ['train'],
#         'gpu': [1],
#         'test_batch': [1407],
#         'full_test': [False],
#         'save_data': [False]
#     }
    configs = {
        'dataset': ['covtype'],
        'epochs': [3, 10],
        'way': [7],
        'train_shot': [1, 5, 10, 100, 300],
        'test_shot': [2],
        'dld': [True],
        'optimizer': ['adam'],
        'lr': [1e-2, 1e-3, 1e-4, 5e-5],
        'initialization': ['xavier_normal'],
        'loss': [nn.MSELoss(reduction='none')],
        'trials': [100],
        'silent': [True],
        'split': ['train'],
        'gpu': [1],
        'test_batch': [2000],
        'full_test': [False],
        'save_data': [False]
    }


    ds_name = configs['dataset'][0]
    if ds_name == 'customer':
        exp_func = run_experiment_customer
    elif ds_name == "covtype":
        exp_func = run_experiment_covtype
    elif ds_name in ['mnist', 'fashion_mnist', 'svhn', 'omniglot']:
        if configs['full_test'][0]:
            exp_func = run_experiment_full_test
        else:
            exp_func = run_experiment
    else:
        raise Exception("Unknown dataset!")

    # Create grid of parameters
    keys, values = zip(*configs.items())
    param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # Create resulting file if necessary
    res_path = f"../../results/02-02-2021/results_{ds_name}_o2m.csv"
    if not os.path.exists(res_path):
        df = pd.DataFrame(columns=configs.keys())
        df.to_csv(res_path, index=False)

    conf_durations = []
    for i, param in enumerate(param_grid):
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

