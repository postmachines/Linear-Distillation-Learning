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


class MLP(nn.Module):
    def __init__(self, n_classes, dim=784):
        super(MLP, self).__init__()

        self.activated_predictor = None
        self.predictors = {}
        self.optimizers = {}
        for c in range(n_classes):
            self.predictors[f'class_{c}'] = nn.Sequential(
                nn.Linear(dim, 50),
                nn.ReLU(),
                nn.Linear(50, 1),
                nn.Sigmoid()
            )
            self.optimizers[f'class_{c}'] = \
                optim.Adam(self.predictors[f'class_{c}'].parameters(),
                           0.001)

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight)

    def activate_predictor(self, class_):
        self.activated_predictor = self.predictors[f'class_{class_}']

    def get_optimizer(self, class_):
        return self.optimizers[f"class_{class_}"]

    def predict(self, x):
        predicts_classes = []
        for predictor in self.predictors:
            predicts_classes.append(self.predictors[predictor](x))
        return np.argmax(predicts_classes)

    def forward(self, next_obs):
        y_score = self.activated_predictor(next_obs)
        return y_score

    def to(self, device):
        super(MLP, self).to(device)
        # Move all predictor networks to the same device
        if torch.cuda.is_available():
            for predictor in self.predictors:
                self.predictors[predictor].to(device)


def train(epoch, model, train_loader):
    model.train()
    criterion = nn.BCELoss()
    for batch_i, (x, y) in enumerate(train_loader):
        x = x.view(x.shape[0], -1).to(device)

        for c in range(10):
            model.activate_predictor(class_=c)
            y_score = model(x).float()
            y_true = y.eq(c).float().to(device)
            loss = criterion(y_score, y_true)
            optimizer = model.get_optimizer(c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if batch_i % 25 == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(msg.format(epoch+1, batch_i, len(train_loader),
                             batch_i/len(train_loader)*100, loss.item()))


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y)  in enumerate(test_loader):
            x = x.view(x.shape[0], -1)
            pred_class = model.predict(x.to(device))
            if pred_class == y.item():
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
    model = MLP(10)
    model.to(device)

    # Dataset of 100 samples (10 per class)
    few_shot_dataset = get_few_shot_mnist(train_loader, shot=10)

    epochs = 20
    for epoch in range(epochs):
        train(epoch, model, few_shot_dataset)
        test(model, test_loader)