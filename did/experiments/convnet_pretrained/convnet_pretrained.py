import numpy as np

import torch
import torch.nn as nn

from did.data.utils import get_few_shot_mnist, get_augmented_images
from did.data import get_n_classes
from did.models import SIMMRESNET18


def train(rnd, loss_func, train_loader, epoch=0):
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

        if batch_i % 100 == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(msg.format(epoch+1, batch_i, len(train_loader),
                         batch_i/len(train_loader)*100, loss.item()))


def test(rnd, test_loader):
    rnd.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(test_loader):
            x = x.squeeze()
            predict_next_state_feature, target_next_state_feature = rnd.predict(x.to(device))
            mses = []
            for predict in predict_next_state_feature:
                mses.append((target_next_state_feature - predict.squeeze()).pow(2).sum(0) / 2)
            class_min_mse = np.argmin(mses)
            if class_min_mse == y.item():
                correct += 1
        acc = correct / (batch_i+1)
        print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i+1, 100. * acc))
    return acc


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    way = 10
    train_shot = 5
    test_shot = 1
    mse_loss = nn.MSELoss(reduction='none')
    accs = []
    trials = 10

    for _ in range(trials):

        data = get_n_classes(way, train_shot, test_shot)

        model = SIMMRESNET18(10)
        model.to(device)

        for sample in data:
            w, h = sample['xs'].shape[-2], sample['xs'].shape[-1]

            x_train = sample['xs'].squeeze().reshape((-1, w, h))
            y_train = [i // train_shot for i in range(train_shot * way)]

            # Shis should be done in preprocessing step
            imgs_aug = []
            y_aug = []
            for i_img in range(x_train.shape[0]):
                img = x_train[i_img].detach().numpy()
                img *= 255

                augmented = get_augmented_images(img, shift=4)
                imgs_aug += augmented
                y_aug += [y_train[i_img]] * len(augmented)

            x_train = np.array(imgs_aug, np.float32) / 255.0
            y_train = np.array(y_aug)

            x_train = x_train.reshape((-1, 1, 28, 28))

            print("x_train: ", x_train.shape)
            print("y_train: ", y_train.shape)

            x_test = sample['xq'].reshape((-1, 1, 28, 28))
            y_test = np.asarray(
                [i // test_shot for i in range(test_shot * way)])

            # break
            x_train = torch.tensor(x_train)
            y_train = torch.tensor(y_train)
            x_test = torch.tensor(x_test)
            y_test = torch.tensor(y_test)

            print("Train: ", x_train.shape, y_train.shape)
            print("Test: ", x_test.shape, y_test.shape)

            inds = np.random.permutation(x_train.shape[0])
            samples_train = list(zip(x_train[inds], y_train[inds]))
            samples_test = list(zip(x_test, y_test))

            train(model, loss_func=mse_loss, train_loader=samples_train)
            accs.append(test(model, samples_test))

    print("Mean accuracy: ", np.mean(accs))
