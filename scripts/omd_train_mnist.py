import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import argparse
import configparser

import torch
import torch.nn as nn
import torchvision

from ldl.models import RNDModel
from utils import preprocess_config


def train(rnd, loss_func, train_loader, epochs, silent=False, device=None, trial=None, log_accuracy=True, test_data_loader=None):
    results_data = []  # trial | split | epoch | sample | predictor | value

    rnd.train()
    for epoch in range(epochs):
        np.random.shuffle(train_loader)
        for batch_i, (x, y) in enumerate(train_loader):
            x = x.squeeze().to(device)
            y = y.to(device)

            # Activate predictor for the needed class
            rnd.activate_predictor(class_=y.item())

            predictor_feature, target_feature = rnd(x)
            loss = loss_func(predictor_feature, target_feature).mean()
            optimizer = rnd.get_optimizer(y.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            results_data.append([trial, "train", epoch, batch_i, y.item(), loss.item()])

            if log_accuracy:
                test_acc = test(rnd, test_data_loader,
                                silent=silent,
                                device=device,
                                test_batch=2000)
                results_data.append(
                    [trial, "test", epoch, batch_i, y.item(), test_acc])

            if batch_i % 100 == 0 and not silent:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                print(msg.format(epoch+1, batch_i, len(train_loader),
                             batch_i/len(train_loader)*100, loss.item()))
    return results_data


def test(rnd, test_loader, silent=False, device='cpu', test_batch=1000):
    rnd.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(test_loader):
            x = x.view(-1, 784).to(device)
            y = y.to(device)

            predict_next_state_feature, target_next_state_feature = rnd.predict(x)

            mses = []
            for predict in predict_next_state_feature:
                mses.append((target_next_state_feature - predict).pow(2).sum(1) / 2)
            mses_tensor = torch.Tensor(10, test_batch).to(device)
            torch.cat(mses, out=mses_tensor)
            mses_tensor = mses_tensor.view(10, test_batch)
            class_min_mse = torch.argmin(mses_tensor, dim=0)
            correct += torch.sum(torch.eq(y, class_min_mse)).item()
        acc = correct / 10_000
        if not silent:
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, 10000, 100. * acc))
    return acc


def preprocess_data(data):
    return data


def run_experiment_full_test(config):
    np.random.seed(2019)
    torch.manual_seed(2019)

    dataset = config['dataset']
    way = config['way']
    train_shot = config['train_shot']
    loss = config['loss']
    trials = config['trials']
    epochs = config['epochs']
    silent = config['silent']
    x_dim = config['x_dim']
    z_dim = config['z_dim']
    c = config['channels']
    optimizer = config['optimizer']
    lr = config['lr']
    initialization = config['initialization']
    gpu = config['gpu']
    test_batch = config['test_batch']
    save_data = config['save_data']

    # Parameters postprocessing
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if loss == 'MSE':
        loss_func = torch.nn.MSELoss(reduction='none')
    else:
        raise ValueError(f"Unknown loss function {loss}")

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

    train_data = {}
    for (x, y) in train_data_loader:
        for i in range(10):
            train_data[i] = x[y==i]

    results_data = [] # trial | split | epoch | sample | predictor | value
    for i_trial in tqdm(range(trials)):
        model = RNDModel(way, in_dim=c*x_dim**2, out_dim=z_dim, opt=optimizer,
                         lr=lr, initialization=initialization)
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

        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)

        x_train = x_train.view(-1, c*x_dim**2)

        inds = np.random.permutation(x_train.shape[0])
        samples_train = list(zip(x_train[inds], y_train[inds]))

        train_trial_data = train(model, loss_func=loss_func, train_loader=samples_train, epochs=epochs,
              silent=silent, device=device, trial=i_trial, log_accuracy=True, test_data_loader=test_data_loader)
        results_data += train_trial_data

        test_acc = test(model, test_data_loader, silent=silent, device=device, test_batch=test_batch)
        results_data.append([i_trial, "test", None, None, None, test_acc])
        accs.append(test_acc)

    # Save results to the file
    if save_data:
        fn_dir = "results"
        fn = f"{fn_dir}/{datetime.datetime.now():%Y-%m-%d_%H:%M}"
        print("Results will be in ", fn)
        if not os.path.exists(fn_dir):
            os.makedirs(fn_dir)
        pd.DataFrame(results_data, columns=["trial", "split", "epoch", "sample", "predictor", "loss/val"]).to_csv(fn+".csv", index=False)
        with open(fn + "_config.txt", "w") as f:
            f.write(str(config))

    return np.mean(accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument("--config", type=str,
                        default="./scripts/mnist.conf",
                        help="Path to the config file.")

    # Run training
    args = vars(parser.parse_args())
    config = configparser.ConfigParser()
    config.read(args['config'])
    config = preprocess_config(config['OMD'])

    from time import time
    time_start = time()
    mean_accuracy = run_experiment_full_test(config)
    print("Elapsed: ", time() - time_start)
    print("Mean accuracy: ", mean_accuracy)
