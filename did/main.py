import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torchvision
from collections import Counter


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
        for c in range(n_classes):
            self.predictors[f'class_{c}'] = nn.Sequential(
                nn.Linear(dim, dim),
            )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight)

        for param in self.target.parameters():
            param.requires_grad = False

    def activate_predictor(self, class_):
        self.activated_predictor = self.predictors[f'class_{class_}']

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

        rnd.activate_predictor(class_=y.item())

        predictor_feature, target_feature = rnd(x)
        loss = mse_loss(predictor_feature, target_feature).mean()
        optimizer = optim.Adam(list(rnd.activated_predictor.parameters()),
                               lr=0.001)
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
    mses = []
    with torch.no_grad():
        for batch_i, (x, y)  in enumerate(test_loader):
            x = x.view(x.shape[0], -1)
            predict_next_state_feature, target_next_state_feature = rnd.predict(x.to(device))
            for predict in predict_next_state_feature:
                mses.append((target_next_state_feature - predict).pow(2).sum(1) / 2)
            class_min_mse = np.argmin(mses)
            if class_min_mse == y.item():
                correct += 1
            mses = []
        print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i+1, 100. * correct / (batch_i+1)))


def get_few_shot_mnist(shot=10):
    few_shot_dataset = []
    class_counter = Counter()
    for batch_i, (x, y) in enumerate(train_loader):
        if class_counter[y.item()] < shot:
            class_counter[y.item()] += 1
            few_shot_dataset.append((x, y))
        if all([class_counter.values() == shot]):
            break
    return few_shot_dataset


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
    rnd = RNDModel(10)
    rnd.to(device)

    # Optimizer and loss
    params = []
    for _, predictor in rnd.predictors.items():
        params += list(predictor.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    mse_loss = nn.MSELoss(reduction='none')

    # Dataset of 100 samples (10 per class)
    few_shot_dataset = get_few_shot_mnist(shot=10)

    epochs = 10
    for epoch in range(epochs):
        train(epoch, rnd, few_shot_dataset)
        test(rnd, test_loader)