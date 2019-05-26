import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from did.data.utils import get_augmented_images
from did.data import get_n_classes
from did.models import SIMMResNet18


def train(rnd, loss_func, train_loader, epoch=0, silent=False):
    for batch_i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        # Activare predictor for the needed class
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
                mses.append((target_next_state_feature - predict.squeeze()).pow(2).sum(0) / 2)
            mses = np.asarray(mses)
            #print("MSEs: ", mses.shape, mses)
            class_min_mse = np.argmin(mses)
            if class_min_mse == y.item():
                correct += 1
        acc = correct / (batch_i+1)
        if not silent:
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i+1, 100. * acc))
    return acc


def augment_data(support):
    """
    Augment data by elementary methods.

    Args:
        support (np.ndarray): data of shape [n_way, n_shot, channels, width, hight]

    Returns (np.ndarray): data of shape [n_augmented, width, height]

    """
    w, h = support.shape[-1], support.shape[-1]
    x_train = sample['xs'].squeeze().reshape((-1, w, h))
    y_train = [i // train_shot for i in range(train_shot * way)]

    # Shis should be done in preprocessing step
    imgs_aug = []
    y_aug = []
    for i_img in range(x_train.shape[0]):
        img = x_train[i_img].detach().numpy()

        augmented = get_augmented_images(img, shift=4, sigma=0.03)
        imgs_aug += augmented
        y_aug += [y_train[i_img]] * len(augmented)

    x_aug = np.array(imgs_aug, np.float32)
    y_aug = np.array(y_aug)
    return x_aug, y_aug


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = {
        'way': 10,
        'train_shot': 5,
        'test_shot': 1,
        'loss': nn.MSELoss(reduction='none'),
        'trials': 10,
        'silent': True
    }
    way = config['way']
    train_shot = config['train_shot']
    test_shot = config['test_shot']
    mse_loss = config['loss']
    trials = config['trials']
    silent = config['silent']

    accs = []
    for _ in tqdm(range(trials)):

        data = get_n_classes(way, train_shot, test_shot)

        model = SIMMResNet18(way)
        model.to(device)

        for sample in data:
            support, query = sample['xs'], sample['xq']
            w, h = support.shape[-2], query.shape[-1]

            x_train = support.squeeze().reshape((-1, w, h))
            y_train = [i // train_shot for i in range(train_shot * way)]
            y_train = np.asarray(y_train)

            # Augmenting
            #x_train, y_train = augment_data(support)
            #x_train = x_train.reshape((-1, w * h))

            x_test = query.reshape((-1, 1, 28, 28))
            y_test = np.asarray(
                [i // test_shot for i in range(test_shot * way)])

            # break
            x_train = torch.tensor(x_train)
            y_train = torch.tensor(y_train)
            x_test = torch.tensor(x_test)
            y_test = torch.tensor(y_test)

            #print("Train: ", x_train.shape, y_train.shape)
            #print("Test: ", x_test.shape, y_test.shape)

            inds = np.random.permutation(x_train.shape[0])
            samples_train = list(zip(x_train[inds], y_train[inds]))
            samples_test = list(zip(x_test, y_test))

            train(model, loss_func=mse_loss, train_loader=samples_train, silent=silent)
            accs.append(test(model, samples_test, silent=silent))

    print("Mean accuracy: ", np.mean(accs))
