import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import argparse
import configparser

import torch
import torchvision

from ldl.models import RNDModel
from utils import preprocess_config, Logger


def train(model, loss_func, train_loader, epochs, logger, silent=False, device=None,
          trial=None, log_test_acc_func=None):
    """
    Train model for one epoch.

    Args:
        model (RNDModel): model object
        loss_func (func): pytorch loss function object
        train_loader (iterable): iterable samples of (x, y) data
        epochs (int): current epoch
        logger (Logger): logger object
        silent (bool): if True print nothing
        device (pt.Device): pytorch device to train on
        trial (int): current trial
        log_test_acc_func (func): function to train accuracy on test

    Returns: None

    """
    model.train()
    for epoch in range(epochs):
        np.random.shuffle(train_loader)
        for batch_i, (x, y) in enumerate(train_loader):
            x = x.squeeze().to(device)
            y = y.to(device)

            # Activate predictor for the needed class
            model.activate_predictor(class_=y.item())

            predictor_feature, target_feature = model(x)
            loss = loss_func(predictor_feature, target_feature).mean()
            optimizer = model.get_optimizer(y.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss on current sample
            logger.log({'trial': trial, 'split': 'train', 'epoch': epoch,
                        'sample': batch_i, 'predictor': y.item(),
                        'value': loss.item()})

            # Accuracy on whole test
            if log_test_acc_func is not None:
                test_acc = log_test_acc_func(model)
                logger.log({'trial': trial, 'split': "test", 'epoch': epoch,
                            'sample': batch_i, 'predictor': y.item(),
                            'value': test_acc})

            if batch_i % 100 == 0 and not silent:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                print(msg.format(epoch+1, batch_i, len(train_loader),
                             batch_i/len(train_loader)*100, loss.item()))


def test(model, data_loader, silent=False, device=None, test_batch=1000):
    """
    Evaluate model on given data.

    Args:
        model (RNDModel): model object
        data_loader (iterable): iterable of (x, y) samples
        silent (bool): if True prints nothing
        device (pt.device): pytorch device to train on
        test_batch (int): size of the test batch

    Returns (float): accuracy value

    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(data_loader):
            x = x.view(-1, 784).to(device)
            y = y.to(device)

            predictor_z, target_z = model.predict(x)

            mses = []
            for predict in predictor_z:
                mses.append((target_z - predict).pow(2).sum(1) / 2)
            mses_tensor = torch.Tensor(10, test_batch).to(device)
            torch.cat(mses, out=mses_tensor)
            mses_tensor = mses_tensor.view(10, test_batch)
            class_min_mse = torch.argmin(mses_tensor, dim=0)
            correct += torch.sum(torch.eq(y, class_min_mse)).item()
        acc = correct / 10_000
        if not silent:
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, 10000, 100. * acc))
    return acc


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
    log_test_accuracy = config['log_test_accuracy']

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

    # Create ready-to-use function to gathering metrics on test
    if log_test_accuracy:
        def foo(model):
            acc = test(model,
                       test_data_loader,
                       device=device,
                       test_batch=test_batch,
                       silent=silent)
            return acc
        log_test_acc_func = foo
    else:
        log_test_acc_func = None

    # Logger
    fn_dir = "results"
    fn = f"{fn_dir}/{datetime.datetime.now():%Y-%m-%d_%H:%M}"
    logger = Logger(filepath=fn,
                    columns=['trial', 'split', 'epoch', 'sample',
                             'predictor', 'value'])

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

        train(model,
              loss_func=loss_func,
              train_loader=samples_train,
              epochs=epochs,
              silent=silent,
              device=device,
              trial=i_trial,
              log_test_acc_func=log_test_acc_func,
              logger=logger)

        # Save accuracy on test only if it is not logged on training
        if log_test_acc_func is None:
            test_acc = test(model,
                            data_loader=test_data_loader,
                            silent=silent,
                            device=device,
                            test_batch=test_batch)
            logger.log({'trial': i_trial, 'split': 'test', 'epoch': None,
                        'sample': None, 'predictor':None, 'value': test_acc})

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
    run_experiment_full_test(config)
    print("Elapsed: ", time() - time_start)