from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from ldl.data import get_episodic_loader
from ldl.models import O2M_svd

def train(o2m, loss_func, way, shot, train_loader, epochs, silent=False, device=None):
    o2m.train()
    np.random.shuffle(train_loader)
    target_feature_dict = {}
    x_data_dict = {}
    x_hat_dict = {}
    for batch_i, (x, y) in enumerate(train_loader):
        x = x.squeeze().to(device)
        y = y.to(device)

#         # Activate predictor for the needed class
#         o2m.activate_predictor(class_=y.item())

        if y.item() not in x_data_dict.keys():
            x_data_dict[y.item()] = []
        if y.item() not in target_feature_dict.keys():
            target_feature_dict[y.item()] = []

        target_feature = o2m(x)

        target_feature_dict[y.item()].append(target_feature)
        x_data_dict[y.item()].append(x)

#     for key in target_feature_dict.keys():
#         target_feature_dict[key] = torch.stack(target_feature_dict[key])
    
    for key in x_data_dict.keys():
        x_data_dict[key] = torch.cat(x_data_dict[key]).reshape(shot, -1)
    for key in target_feature_dict.keys():
        target_feature_dict[key] = torch.cat(target_feature_dict[key]).reshape(shot, -1)
    
#     print(f'single class size is {x_data_dict[0].shape}')
    x_data_dict_merged = torch.cat([x_data_dict[key] for key in range(way)], 1).reshape(shot, -1)
#     print(f'all data merged size is {x_data_dict_merged.shape}')   
    
    target_feature_dict_merged = torch.cat([target_feature_dict[key] for key in range(way)], 1).reshape(shot, -1)
#     print(f'all target feature merged size is {target_feature_dict_merged.shape}')   

#     target_feature_dict_merged = torch.cat(torch.tensor([target_feature_dict[k] for k in range(min(target_feature_dict.keys()), 
#                                                                                     max(target_feature_dict.keys())+1)]))

    # calculate the economy SVD for the data matrix A
    U,S,Vt = torch.linalg.svd(x_data_dict_merged, full_matrices=False)

    # solve Ax = b for the best possible approximate solution in terms of least squares
    x_hat = Vt.T @ torch.linalg.inv(torch.diag(S)) @ U.T @ target_feature_dict_merged
#     print(x_hat.shape)
    x_hat_split = torch.split(x_hat, x_hat.shape[0]//way, dim=0)
    
    for i in range(way):
        o2m.predictors[f'class_{i}'] = torch.split(x_hat_split[i], x_hat_split[i].shape[1]//way, dim=1)[i]
#         print(o2m.predictors[f'class_{i}'].shape)

def test(o2m, test_loader, silent=False, device='cpu'):
    o2m.eval()
    correct = 0
    with torch.no_grad():
        n = len(test_loader)
        for batch_i, (x, y) in enumerate(test_loader):
            x = x.squeeze()
            predict_next_state_feature, target_next_state_feature = o2m.predict(x.to(device))
            mses = []
            for predict in predict_next_state_feature:
                mses.append((target_next_state_feature - predict).pow(2).sum(0) / 2)
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
    initialization = config['initialization']
    gpu = config['gpu']
    table_dataset = config['table_dataset']
    #dld = config['dld']

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
        model = O2M_svd(way, in_dim=in_dim, out_dim=z_dim,initialization=initialization)#, dld=dld)
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

            train(model, loss_func=mse_loss, way=way, shot=train_shot, train_loader=samples_train, epochs=epochs,
                  silent=silent, device=device)
            accs.append(test(model, samples_test, silent=silent, device=device))

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
