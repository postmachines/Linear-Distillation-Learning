import os
from time import time
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from train import run_experiment
from train_full_test import run_experiment_full_test


if __name__ == "__main__":
    print("GPU available: ", torch.cuda.is_available())

    configs = {
        'dataset': ['omniglot'],
        'epochs': [10, 3],
        'way': [3, 5, 10],
        'train_shot': [1, 3, 5, 10],
        'test_shot': [1],
        'x_dim': [28], # ATTENTION: Due to the cached nature of dataloader this parameter should be set in signle value per run
        'z_dim': [784, 2000],
        'optimizer': ['adam'],
        'lr': [1e-3, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7],
        'initialization': ['xavier_normal'],
        'channels': [1],
        'loss': [nn.MSELoss(reduction='none')],
        'trials': [100],
        'silent': [True],
        'split': ['test'],
        'in_alphabet': [False],
        'add_rotations': [True],
        'gpu': [1],
        'test_batch': [2000],
        'full_test': [False],
        'save_data': [False]
    }

    if configs['full_test'][0]:
        experiment_func = run_experiment_full_test
    else:
        experiment_func = run_experiment

    # Create grid of parameters
    keys, values = zip(*configs.items())
    param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create resulting file if necessary
    res_path = "did/experiments/base/results_omni.csv"
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
        mean_accuracy = experiment_func(config=param)
        conf_durations.append(time() - time_start)
        df = pd.read_csv(res_path)
        df = df.append(pd.Series({**param, **{'accuracy': mean_accuracy,
                                              'duration_sec': conf_durations[-1]}}), ignore_index=True)
        df.to_csv(res_path, index=False)

