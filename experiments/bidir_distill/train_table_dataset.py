import os
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision

from ldl.models import BidirDistill
from ldl.data import get_episodic_loader

def train_target_epoch(epoch, model, data_loader, loss_func, device, trial, silent=True):
    """
    Target target network for single epoch.

    Args:
        epoch (int): current epoch
        model (BidirDistill): BidirDistill model object
        data_loader (itertable): data loader
        loss_func (func): pytorch loss function
        device (torch.Device): device on which to train
        trial (int): number of trial
        silent (bool): if True print nothing.

    Returns: None

    """
    results_data = []
    n = len(data_loader)
    for batch_i, (x, y) in enumerate(data_loader):
        x = x.view(1, x.shape[0]).to(device)
        y = y.to(device)

        # Activate predictor corresponding to the true label
        model.activate_predictor(class_=y.item())

        predictor_z, target_z = model(x)
        loss = loss_func(target_z, predictor_z).mean()

        # Update target weights
        optimizer = model.optimizer_target
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not silent and batch_i % 1000 == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(msg.format(epoch+1, batch_i, n, batch_i/n*100, loss.item()))

        # Logging info
        results_data.append(
            [trial, "train", epoch, batch_i, 'Teacher', loss.item()])

    return results_data


def test_target(model, test_loader, device, silent=True):
    """
    Test target network
    Args:
        model (LDL object): object of model to train
        test_loader (iterable): data loader
        device (torch.Device): device to move data to
        silent (bool): if True prints nothing

    Returns: None

    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(test_loader):
            x = x.view(x.shape[0], -1)
            predict_next_state_feature, target_next_state_feature = model.predict(x.to(device))
            mses = []
            for predict in predict_next_state_feature:
                mses.append((predict - target_next_state_feature).pow(2).sum(1) / 2)
            class_min_mse = np.argmin(mses)
            if class_min_mse == y.item():
                correct += 1
        if not silent:
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i+1, 100. * correct / (batch_i+1)))


def train_predictors_epoch(model, data_loader, loss_func, device, trial,
                           epoch, silent=True, log_accuracy=True, test_data_loader=None):
    """
    Train predictors networks for single epoch.

    Args:
        model (BidirDistill): object of model to train
        data_loader (iterable): data loader of (x, y) samples
        loss_func (func): torch loss function
        device (torch.Device): device to move data to
        trial (int): number of trial (for logging)
        epoch (int): number of current epoch (for logging)
        silent (bool): if true outputs nothing

    Returns (list): list of lists of [trial_n, 'train', epoch_n, sample_n, predictor_n, loss]

    """

    n = len(data_loader)
    results_data = []  # trial | split | epoch | sample | predictor | value
    for batch_i, (x, y) in enumerate(data_loader):
        x = x.view(1, x.shape[0]).to(device)
        y = y.to(device)

        # Activate corresponding predictor and get corresponding optimizer
        model.activate_predictor(class_=y.item())
        optimizer = model.get_optimizer(y.item())

        # Forward sample and calucate loss
        predictor_z, target_z = model(x)
        loss = loss_func(predictor_z, target_z).mean()

        # Backpropagate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging info
        results_data.append(
            [trial, "train", epoch, batch_i, f"Student {y.item()}", loss.item()])

        if log_accuracy:
            acc = test_predictors(model, test_data_loader, device,
                                  test_batch=2000,
                                  silent=True)
            results_data.append(
                [trial, "test", epoch, batch_i, "Predictors", acc])

        if not silent and batch_i % 100 == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(msg.format(epoch+1, batch_i, n, batch_i/n*100, loss.item()))

    return results_data


def test_predictors(model, data_loader, device, test_batch=1000, silent=True, x_dim=41, way=2):
    """
    Get accuracy of the model's predictors.

    Args:
        model (BidirDistill): object of model to get predicts from
        data_loader (iterable): data loader of form (x, y) samples
        device (torch.Device): device to move data to
        test_batch (int): batch size while testing
        silent (bool): if True outputs nothing

    Returns:

    """
    model.eval()
    correct = 0
    with torch.no_grad():
        n = len(data_loader)
        
        for batch_i, (x, y) in enumerate(data_loader):
            x = x.view(-1, x_dim).to(device)
            y = y.to(device)

            predict_next_state_feature, target_next_state_feature = model.predict(x)

            mses = []
            for predict in predict_next_state_feature:
                mses.append(
                    (target_next_state_feature - predict).pow(2).sum(1) / 2)
            mses_tensor = torch.Tensor(way, test_batch).to(device)
            torch.cat(mses, out=mses_tensor)
            mses_tensor = mses_tensor.view(way, test_batch)
            class_min_mse = torch.argmin(mses_tensor, dim=0)
            correct += torch.sum(torch.eq(y, class_min_mse)).item()
        acc = correct / n
        if not silent:
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, n, 100. * acc))
    return acc


def train(model, loss_func, train_loader, epochs, device, trial, silent, test_data_loader=None, log_accuracy=False):
    """
    Train BidirDistill for given number of epochs.

    Args:
        model (BidirDistill): object of model to train
        loss_func (func): torch loss function
        train_loader (iterable): data loader
        epochs (int): number epochs to train
        device (torch.Device): device to move data to
        trial (int): number of trial (for logging)
        silent (bool): if True print nothing

    Returns:

    """
    results_data = []  # trial | split | epoch | sample | predictor | value
    model.train()
    for epoch in range(epochs):
        np.random.shuffle(train_loader)

        train_data = []

        # (1) Train target
        train_data += train_target_epoch(model=model,
                           data_loader=train_loader,
                           loss_func=loss_func,
                           device=device,
                           epoch=epoch,
                           trial=trial,
                           silent=silent)
        
        # (2) Train predictors
        train_data += train_predictors_epoch(model=model,
                                            data_loader=train_loader,
                                            loss_func=loss_func,
                                            device=device, trial=trial,
                                            epoch=epoch,
                                            silent=silent,
                                             log_accuracy=log_accuracy,
                                             test_data_loader=test_data_loader)
        results_data += train_data

    return results_data


def preprocess_data(data):
    return data


def run_experiment(config):
    np.random.seed(2019)
    torch.manual_seed(2019)

    dataset = config['dataset']
    train_shot = config['train_shot']
    test_shot = config['test_shot']
    x_dim = config['x_dim']
    z_dim = config['z_dim']
    way = config['way']
    mse_loss = config['loss']
    trials = config['trials']
    epochs = config['epochs']
    test_batch = config['test_batch']
    silent = config['silent']
    split = config['split']
    lr_predictor = config['lr_predictor']
    lr_target = config['lr_target']
    gpu = config['gpu']

    #dld = config['dld']

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    accs = []
    dataloader = get_episodic_loader(dataset, way, train_shot, test_shot, x_dim,
                                     split=split)

    results_data = []  # trial | split | epoch | sample | predictor | value
    for i_trial in tqdm(range(trials)):

        for sample in dataloader:
            x_train = sample['xs'].reshape((-1, x_dim))
            y_train = np.asarray(
                [i // train_shot for i in range(train_shot * way)])
            x_test = sample['xq'].reshape((-1, x_dim))
            y_test = np.asarray(
                [i // test_shot for i in range(test_shot * way)])

            x_train = preprocess_data(x_train)
            x_test = preprocess_data(x_test)

            x_train = torch.tensor(x_train)
            y_train = torch.tensor(y_train)
            x_test = torch.tensor(x_test)
            y_test = torch.tensor(y_test)

            # print("Train: ", x_train.shape, y_train.shape)
            # print("Test: ", x_test.shape, y_test.shape)

            inds = np.random.permutation(x_train.shape[0])
            samples_train = list(zip(x_train[inds], y_train[inds]))
            samples_test = list(zip(x_test, y_test))
            
            model = BidirDistill(n_classes=way,
                    in_dim=x_dim,
                    out_dim=z_dim,
                    lr_predictor=lr_predictor,
                    lr_target=lr_target)
            model.to(device)

            train(model, loss_func=mse_loss, train_loader=samples_train, epochs=epochs,
                  silent=silent, device=device, trial=i_trial, log_accuracy=False)
#             results_data += train_trial_data
            test_acc = test_predictors(model, samples_test, silent=silent, 
                                       device=device, test_batch=test_batch, 
                                       x_dim=x_dim, way=way)
            accs.append(test_acc)

#             results_data.append([i_trial, "test", None, None, None, test_acc])
#     # Save results to the file
#     if save_data:
#         fn_dir = "results"
#         fn = f"{fn_dir}/{datetime.datetime.now():%Y-%m-%d_%H:%M}"
#         if not os.path.exists(fn_dir):
#             os.makedirs(fn_dir)
#         cols = ["trial", "split", "epoch", "sample", "predictor", "loss/val"]
#         pd.DataFrame(results_data, columns=cols).to_csv(fn + ".csv",
#                                                         index=False)
#         with open(fn + "_config.txt", "w") as f:
#             f.write(str(config))

    return np.mean(accs)


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
    x_dim = config['x_dim']
    z_dim = config['z_dim']
    c = config['channels']
    lr_predictor = config['lr_predictor']
    lr_target = config['lr_target']
    gpu = config['gpu']
    test_batch = config['test_batch']
    save_data = config['save_data']

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

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
        for i in range(2):
            train_data[i] = x[y==i]

    results_data = [] # trial | split | epoch | sample | predictor | value
    for i_trial in tqdm(range(trials)):
        model = BidirDistill(n_classes=way,
                    in_dim=x_dim**2,
                    out_dim=z_dim,
                    lr_predictor=lr_predictor,
                    lr_target=lr_target)
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

        # Convert data to tensors
        x_train = torch.tensor(x_train).view(-1, c*x_dim**2)
        y_train = torch.tensor(y_train)

        # Shuffle data
        inds = np.random.permutation(x_train.shape[0])
        samples_train = list(zip(x_train[inds], y_train[inds]))

        # Train target + predictors for epoch
        train_trial_data = train(model=model,
                                 loss_func=mse_loss,
                                 train_loader=samples_train,
                                 epochs=epochs,
                                 device=device,
                                 trial=i_trial,
                                 silent=silent, 
                                 test_data_loader=test_data_loader)
        results_data += train_trial_data

        # Check accuracy
        test_acc = test_predictors(model=model,
                                   data_loader=test_data_loader,
                                   device=device,
                                   test_batch=test_batch,
                                   silent=silent)
        if False:
            results_data.append([i_trial, "test", None, None, None, test_acc])
        accs.append(test_acc)

    # Save results to the file
    if save_data:
        fn_dir = "results"
        fn = f"{fn_dir}/{datetime.datetime.now():%Y-%m-%d_%H:%M}"
        print("Data will be saved into ", fn)
        if not os.path.exists(fn_dir):
            os.makedirs(fn_dir)
        cols = ["trial", "split", "epoch", "sample", "predictor", "loss/val"]
        pd.DataFrame(results_data, columns=cols).to_csv(fn+".csv", index=False)
        with open(fn + "_config.txt", "w") as f:
            f.write(str(config))

    return np.mean(accs)


if __name__ == "__main__":
    config = {
        'dataset': 'customer',
        'way': 2,
        'train_shot': 1,
        'test_shot': 1,
        'loss': nn.MSELoss(reduction='none'),
        'epochs': 1,
        'trials': 1,
        'silent': True,
        'split': 'test',
        'lr_predictor': 1e-3,
        'lr_target': 1e-3,
        'gpu': 1,
        'test_batch': 2000,
        'save_data': True,
        'log_accuracy': True
    }

    from time import time
    time_start = time()
    mean_accuracy = run_experiment_full_test(config)
    print("Elapsed: ", time() - time_start)
    print("Mean accuracy: ", mean_accuracy)

