import datetime
import os
import itertools
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


from ldl.data import get_episodic_loader


def preprocess_data(data):
    return data


def run_experiment(config):
    np.random.seed(2019)

    dataset = config['dataset']
    way = config['way']
    train_shot = config['train_shot']
    test_shot = config['test_shot']
    trials = config['trials']
    split = config['split']
    x_dim = config['x_dim']
    c = config['channels']
    add_rotations = config['add_rotations']
    max_depth = config['max_depth']
    in_alphabet = config['in_alphabet']
    #dld = config['dld']


    accs = []
    dataloader = get_episodic_loader(dataset, way, train_shot, test_shot, x_dim,
                                     split=split,
                                     add_rotations=add_rotations,
                                     in_alphabet=in_alphabet)

    for i_trial in tqdm(range(trials)):
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=0)

        for sample in dataloader:
            x_train = sample['xs'].reshape((-1, c*x_dim**2))
            y_train = np.asarray(
                [i // train_shot for i in range(train_shot * way)])
            x_test = sample['xq'].reshape((-1, c*x_dim**2))
            y_test = np.asarray(
                [i // test_shot for i in range(test_shot * way)])

            x_train = preprocess_data(x_train)
            x_test = preprocess_data(x_test)
#             print(f'X train shape {x_train.shape}')

            inds = np.random.permutation(x_train.shape[0])
            samples_train = list(zip(x_train[inds], y_train[inds]))
#             print(f'samples_train shape {len(samples_train)}')

            samples_test = list(zip(x_test, y_test))
            
            model.fit(x_train[inds], y_train[inds])

            y_pred = model.predict(x_test)
            accs.append(accuracy_score(y_test, y_pred))

    return np.mean(accs)


def run_experiment_full_test(config):
    pass


if __name__ == "__main__":

    configs = {
        'dataset': ['fashion_mnist'],
        'way': [10],
        'train_shot': [1, 10, 50, 100, 200, 300],
        'test_shot': [1],
        'max_depth': [1, 3, 5, 7, 10, 20, 30, 50, 100],
        'x_dim': [28],
        'channels': [1],
        'trials': [100],
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
    ds_name = configs['dataset'][0]
    res_path = f"../../results/02-02-2021/results_{ds_name}_tree.csv"
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