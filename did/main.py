import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torchvision

from data.utils import get_few_shot_mnist
from data import load_omniglot, get_n_classes


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class RNDModel(nn.Module):
    def __init__(self, n_classes, dim=784):
        super(RNDModel, self).__init__()

        self.activated_predictor = None
        self.target = nn.Sequential(nn.Linear(dim, dim))
        self.predictors = {}
        self.optimizers = {}
        for c in range(n_classes):
            self.predictors[f'class_{c}'] = nn.Sequential(
                nn.Linear(dim, dim)
            )
            self.optimizers[f'class_{c}'] = \
                optim.Adam(self.predictors[f'class_{c}'].parameters(),
                           0.001)

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight)

        for param in self.target.parameters():
            param.requires_grad = False

    def activate_predictor(self, class_):
        self.activated_predictor = self.predictors[f'class_{class_}']

    def get_optimizer(self, class_i):
        return self.optimizers[f"class_{class_i}"]

    def predict(self, next_obs):
        predict_features = []
        target_feature = self.target(next_obs)
        for predictor in self.predictors:
            predict_features.append(self.predictors[predictor](next_obs))
        return predict_features, target_feature

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.activated_predictor(next_obs)
        return predict_feature, target_feature

    def to(self, device):
        super(RNDModel, self).to(device)
        # Move all predictor networks to the same device
        if torch.cuda.is_available():
            for predictor in self.predictors:
                self.predictors[predictor].to(device)


def train(rnd, loss_func, train_loader, epoch=0):
    for batch_i, (x, y) in enumerate(train_loader):
        x = x.squeeze().to(device)
        y = y.to(device)

        rnd.activate_predictor(class_=y.item())

        predictor_feature, target_feature = rnd(x)
        loss = loss_func(predictor_feature, target_feature).mean()
        optimizer = rnd.get_optimizer(y.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_i % 25 == 0:
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
                mses.append((target_next_state_feature - predict).pow(2).sum(0) / 2)
            class_min_mse = np.argmin(mses)
            if class_min_mse == y.item():
                correct += 1
        acc = correct / (batch_i+1)
        print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i+1, 100. * acc))
    return acc


if __name__ == "__main__":
    torch.manual_seed(2019)

    way = 5
    train_shot = 5
    test_shot = 1
    mse_loss = nn.MSELoss(reduction='none')
    accs = []

    for _ in range(100):

        data = get_n_classes(way, train_shot, test_shot)

        model = RNDModel(10)
        model.to(device)

        for sample in data:
            x_train = sample['xs'].reshape((-1, 28*28))
            y_train = np.asarray([i//train_shot for i in range(train_shot*way)])
            x_test = sample['xq'].reshape((-1, 28*28))
            y_test = np.asarray([i//test_shot for i in range(test_shot*way)])

            x_train = torch.tensor(x_train)
            y_train = torch.tensor(y_train)
            x_test = torch.tensor(x_test)
            y_test = torch.tensor(y_test)

            #print("Train: ", x_train.shape, y_train.shape)
            #print("Test: ", x_test.shape, y_test.shape)

            inds = np.random.permutation(x_train.shape[0])
            samples_train = list(zip(x_train[inds], y_train[inds]))
            samples_test = list(zip(x_test, y_test))

            train(model, loss_func=mse_loss, train_loader=samples_train)
            accs.append(test(model, samples_test))

    print("Mean accuracy: ", np.mean(accs))