from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from ldl.data import get_episodic_loader
from ldl.models import RNDModel


def train(rnd, loss_func, train_loader, epochs, silent=False, device=None):
    rnd.train()
    np.random.shuffle(train_loader)
    target_feature_dict = {}
    x_data_dict = {}
    x_hat_dict = {}
    for batch_i, (x, y) in enumerate(train_loader):
        x = x.squeeze().to(device)
        y = y.to(device)

        # Activate predictor for the needed class
        rnd.activate_predictor(class_=y.item())

        if y.item() not in x_data_dict.keys():
            x_data_dict[y.item()] = []
        if y.item() not in target_feature_dict.keys():
            target_feature_dict[y.item()] = []

        _, target_feature = rnd(x)

        target_feature_dict[y.item()].append(target_feature)
        x_data_dict[y.item()].append(x)

    for key in target_feature_dict.keys():
        target_feature_dict[key] = torch.stack(target_feature_dict[key])
    for key in x_data_dict.keys():
        x_data_dict[key] = torch.stack(x_data_dict[key])
        # calculate the economy SVD for the data matrix A
        U,S,Vt = torch.linalg.svd(x_data_dict[key], full_matrices=False)

        # solve Ax = b for the best possible approximate solution in terms of least squares
        x_hat = Vt.T @ torch.linalg.inv(torch.diag(S)) @ U.T @ target_feature_dict[key]
        x_hat_dict[key] = x_hat

    return x_hat_dict


def test(rnd, test_loader, x_hat_dict, silent=False, device='cpu'):
    rnd.eval()
    correct = 0
    with torch.no_grad():
        n = len(test_loader)
        
        for batch_i, (x, y) in enumerate(test_loader):
            x = x.squeeze().to(device)
            _, target_next_state_feature = rnd.predict(x)
#             x = torch.stack([x])
            y = y.to(device)
            mses = []
            for k, x_hat in sorted(x_hat_dict.items()):
#                 print(k)
                test_prediction = (x @ x_hat).squeeze() # хз
#                 print(test_prediction.shape)
#                 print(f'target {target_next_state_feature.shape}')
                mses.append((target_next_state_feature - test_prediction).pow(2).sum(0) / 2)
            class_min_mse = np.argmin(mses)
            if class_min_mse == y.item():
                correct += 1
        acc = correct / n
        if not silent:
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, n, 100. * acc))
    return acc


def preprocess_data(data):
    return data


def run_experiment(config):
    np.random.seed(2019)
    torch.manual_seed(2019)

    dataset = config['dataset']
    way = config['way']
    train_shot = config['train_shot']
    test_shot = config['test_shot']
    mse_loss = config['loss']
    trials = config['trials']
    epochs = config['epochs']
    silent = config['silent']
    split = config['split']
    add_rotations = config['add_rotations']
    in_alphabet = config['in_alphabet']
    x_dim = config['x_dim']
    z_dim = config['z_dim']
    c = config['channels']
    optimizer = config['optimizer']
    lr = config['lr']
    initialization = config['initialization']
    gpu = config['gpu']
    #dld = config['dld']
    table_dataset = config['table_dataset']
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    accs = []
    dataloader = get_episodic_loader(dataset, way, train_shot, test_shot, x_dim,
                                     split=split,
                                     add_rotations=add_rotations,
                                     in_alphabet=in_alphabet)
    if table_dataset:
        in_dim = x_dim
    else:
        in_dim = c*x_dim**2
        
    for _ in tqdm(range(trials)):
        model = RNDModel(way, in_dim=in_dim, out_dim=z_dim, opt=optimizer,
                         lr=lr, initialization=initialization)#, dld=dld)
        model.to(device)

        for sample in dataloader:
            x_train = sample['xs'].reshape((-1, in_dim))
            y_train = np.asarray(
                [i // train_shot for i in range(train_shot * way)])
            x_test = sample['xq'].reshape((-1, in_dim))
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

            x_hat_dict = train(model, loss_func=mse_loss, train_loader=samples_train, epochs=epochs,
                  silent=silent, device=device)
            accs.append(test(model, samples_test, x_hat_dict, silent=silent, device=device))

    return np.mean(accs)


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = {
        'dataset': 'mnist',
        'way': 10,
        'train_shot': 5,
        'test_shot': 1,
        'loss': nn.MSELoss(reduction='none'),
        'epochs': 1,
        'trials': 10,
        'silent': True,
        'split': 'test',
        'in_alphabet': False,
        'add_rotations': False,
        'x_dim': 28,
        'z_dim': 784,
        'initialization': 'xavier_normal',
        'optimizer': 'adam',
        'lr': 0.001,
        'channels': 1,
        'gpu': 1
    }

    mean_accuracy = run_experiment(config)
    print("Mean accuracy: ", mean_accuracy)
