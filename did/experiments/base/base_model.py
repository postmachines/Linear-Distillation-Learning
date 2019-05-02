from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from did.data import get_episodic_loader
from did.models import RNDModel


def train(rnd, loss_func, train_loader, epoch=0, silent=False):
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

        if batch_i % 100 == 0 and not silent:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(msg.format(epoch+1, batch_i, len(train_loader),
                         batch_i/len(train_loader)*100, loss.item()))


def test(rnd, test_loader, silent=False):
    rnd.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(test_loader):
            x = x.squeeze()
            predict_next_state_feature, target_next_state_feature = rnd.predict(x.to(device))
            mses = []
            for predict in predict_next_state_feature:
                mses.append((target_next_state_feature - predict).pow(2).sum(0) / 2)
            class_min_mse = np.argmin(mses)
            if class_min_mse == y.item():
                correct += 1
        acc = correct / (batch_i+1)
        if not silent:
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i+1, 100. * acc))
    return acc


def preprocess_data(data):
    return data


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = {
        'way': 10,
        'train_shot': 5,
        'test_shot': 1,
        'loss': nn.MSELoss(reduction='none'),
        'trials': 100,
        'silent': True,
        'split': 'test',
        'add_rotations': False
    }
    way = config['way']
    train_shot = config['train_shot']
    test_shot = config['test_shot']
    mse_loss = config['loss']
    trials = config['trials']
    silent = config['silent']
    split = config['split']
    add_rotations = config['add_rotations']

    accs = []
    for _ in tqdm(range(trials)):

        dataloader = get_episodic_loader(way, train_shot, test_shot,
                                         split=split,
                                         add_rotations=add_rotations)

        model = RNDModel(way)
        model.to(device)

        for sample in dataloader:
            x_train = sample['xs'].reshape((-1, 28 * 28))
            y_train = np.asarray(
                [i // train_shot for i in range(train_shot * way)])
            x_test = sample['xq'].reshape((-1, 28 * 28))
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

            train(model, loss_func=mse_loss, train_loader=samples_train,
                  silent=silent)
            accs.append(test(model, samples_test, silent=silent))

    print("Mean accuracy: ", np.mean(accs))


