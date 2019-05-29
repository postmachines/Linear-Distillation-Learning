import datetime
import os
import itertools
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision
import torch.optim as optim

from models import TinyCNN


def train(model, loss_func, train_loader, epochs, optimizer, lr, batch_size, silent=False, device=None, trial=None):
    results_data = []  # trial | split | epoch | sample | value

    opt = optimizer.lower()
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr)
    elif opt == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")

    def batch_loader(samples, batch_size):
        np.random.shuffle(samples)
        for batch_i in range(len(samples) // batch_size + 1):
            batch_samples = samples[batch_i*batch_size:(batch_i+1)*batch_size]
            x = torch.stack([s[0] for s in batch_samples])
            y = torch.tensor([s[1] for s in batch_samples])
            yield x, y

    model.train()
    for epoch in range(epochs):
        batch_load = batch_loader(train_loader, batch_size)
        for batch_i, (x, y) in enumerate(batch_load):
            x = x.view(-1, 1, 28, 28).to(device)
            y = y.to(device)

            y_score = model(x)
            loss = loss_func(y_score, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            results_data.append([trial, "train", epoch, batch_i,loss.item()])

            if batch_i % 100 == 0 and not silent:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                print(msg.format(epoch+1, batch_i, len(train_loader),
                             batch_i/len(train_loader)*100, loss.item()))
    return results_data


def test(model, test_loader, silent=False, device='cpu'):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(test_loader):
            x = x.view(-1, 1, 28, 28).to(device)
            y = y.to(device)

            y_score = model(x)
            vals, inds = torch.max(y_score, 1)
            correct += torch.sum(torch.eq(y, inds)).item()
        acc = correct / 10_000
        if not silent:
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, 10000, 100. * acc))
    return acc


def run_experiment(config):
    pass


def run_experiment_full_test(config):
    np.random.seed(2019)
    torch.manual_seed(2019)

    dataset = config['dataset']
    way = config['way']
    train_shot = config['train_shot']
    mse_loss = config['loss']
    trials = config['trials']
    epochs = config['epochs']
    silent = config['silent']
    batch_size = config['batch_size']
    x_dim = config['x_dim']
    c = config['channels']
    optimizer = config['optimizer']
    lr = config['lr']
    nonlinearity = config['nonlinearity']
    gpu = config['gpu']
    test_batch = config['test_batch']

    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    accs = []
    train_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/MNIST/', train=True,
                                   download=True,
                                   transform=torchvision.transforms.ToTensor(),
                                   ),
        batch_size=60000, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/MNIST/', train=False,
                                   download=True,
                                   transform=torchvision.transforms.ToTensor()),
        batch_size=test_batch, shuffle=True
    )

    # Gather training data by classes
    train_data = {}
    for (x, y) in train_data_loader:
        for i in range(10):
            train_data[i] = x[y == i]

    results_data = []  # trial | split | epoch | sample | value
    for i_trial in tqdm(range(trials)):
        model = TinyCNN(way)
        model.to(device)

        # Select random shot
        x_train = []
        y_train = []
        for i in range(10):
            inds = np.arange(train_data[i].shape[0])
            np.random.shuffle(inds)
            x_train.append(train_data[i][inds[:train_shot]])
            y_train += [i] * train_shot
        x_train = np.vstack(x_train)
        y_train = np.asarray(y_train).reshape((-1, 1))

        # Convert to tensor
        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)
        x_train = x_train.view(-1, c * x_dim ** 2)

        samples_train = list(zip(x_train, y_train))
        train_trial_data = train(model, loss_func=mse_loss,
                                 optimizer=optimizer, lr=lr,
                                 train_loader=samples_train, epochs=epochs,
                                 batch_size=batch_size, silent=silent,
                                 device=device, trial=i_trial)
        results_data += train_trial_data

        test_acc = test(model, test_data_loader, silent=silent, device=device)
        results_data.append([i_trial, "test", None, None, test_acc])
        accs.append(test_acc)

    # Save results to the file
    fn_dir = "results"
    fn = f"{fn_dir}/{datetime.datetime.now():%Y-%m-%d_%H:%M}"
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)
    pd.DataFrame(results_data,
                 columns=["trial", "split", "epoch", "sample",
                          "loss/val"]).to_csv(fn + ".csv", index=False)
    with open(fn + "_config.txt", "w") as f:
        f.write(str(config))

    return np.mean(accs)


if __name__ == "__main__":
    print("GPU available: ", torch.cuda.is_available())

    configs = {
        'dataset': ['mnist'],
        'epochs': [3, 10],
        'way': [10],
        'train_shot': [1, 5, 10, 50, 100, 200],
        'test_shot': [1],
        'x_dim': [28],
        'hidden_layers': [1, 2],
        'nonlinearity': ['relu'],
        'optimizer': ['adam'],
        'lr': [1e-3],
        'channels': [1],
        'loss': [nn.CrossEntropyLoss()],
        'trials': [100],
        'batch_size': [32],
        'silent': [True],
        'split': ['test'],
        'in_alphabet': [False],
        'add_rotations': [True],
        'gpu': [1],
        'test_batch': [2000],
        'full_test': [True],
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
    res_path = f"experiments/baselines/results_cnn_{ds_name}.csv"
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