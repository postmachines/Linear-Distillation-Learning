import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torchvision

from data.utils import get_few_shot_mnist

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class RNDModel(nn.Module):
    def __init__(self, n_classes, input_dim=784, output_dim=784,
                 nonlinearity='relu'):
        super(RNDModel, self).__init__()

        self.activated_predictor = None
        self.target = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.predictors = {}
        self.optimizers = {}
        nonlinearity = nonlinearity.lower()
        if nonlinearity == 'relu':
            nonlin = nn.ReLU
        elif nonlinearity  == 'sigmoid':
            nonlin = nn.Sigmoid
        elif nonlinearity == 'prelu':
            nonlin = nn.PReLU
        for c in range(n_classes):
            self.predictors[f'class_{c}'] = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nonlin(),
                nn.Linear(output_dim, output_dim),
                nonlin()
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


def train(epoch, rnd, train_loader):
    for batch_i, (x, y) in enumerate(train_loader):
        x = x.view(x.shape[0], -1).to(device)
        y = y.to(device)

        for c in range(10):
            rnd.activate_predictor(class_=y.item())
            predictor_feature, target_feature = rnd(x)
            if c == y.item():
                # Reduce MSE between target and predictor
                loss = mse_loss(predictor_feature, target_feature).mean()
                optimizer = rnd.get_optimizer(y.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # Increase MSE between target and predictor
                loss_neg = -mse_loss(predictor_feature, target_feature).mean()
                optimizer = optim.Adam(rnd.predictors[f'class_{c}'].parameters(),
                                       lr=0.0001)
                optimizer.zero_grad()
                loss_neg.backward()
                optimizer.step()

        if batch_i % 25 == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(msg.format(epoch+1, batch_i, len(train_loader),
                         batch_i/len(train_loader)*100, loss.item()))


def test(rnd, test_loader):
    rnd.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y)  in enumerate(test_loader):
            x = x.view(x.shape[0], -1)
            predict_next_state_feature, target_next_state_feature = rnd.predict(x.to(device))
            mses = []
            for predict in predict_next_state_feature:
                mses.append((target_next_state_feature - predict).pow(2).sum(1) / 2)
            class_min_mse = np.argmin(mses)
            if class_min_mse == y.item():
                correct += 1
        print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i+1, 100. * correct / (batch_i+1)))


if __name__ == "__main__":
    torch.manual_seed(2019)

    # Load data
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/MNIST/', train=True, download=True,
                       transform=torchvision.transforms.ToTensor(),
                       ),
        batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/MNIST/', train=False, download=True,
                        transform=torchvision.transforms.ToTensor(),
                       ),
        batch_size=1, shuffle=True)

    # Random Network Distillation
    rnd = RNDModel(10, input_dim=784,
                   output_dim=784,
                   nonlinearity='sigmoid')
    rnd.to(device)

    # Loss
    mse_loss = nn.MSELoss(reduction='none')

    # Dataset of 100 samples (10 per class)
    few_shot_dataset = get_few_shot_mnist(train_loader, shot=10)

    epochs = 10
    for epoch in range(epochs):
        train(epoch, rnd, few_shot_dataset)
        test(rnd, test_loader)