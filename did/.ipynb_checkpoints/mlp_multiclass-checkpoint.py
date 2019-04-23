import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

from data.utils import get_few_shot_mnist

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class MLP(nn.Module):
    def __init__(self, x_dim, y_dim, hidden=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(x_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, y_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def train(epoch, model, data_loader):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 0.01)
    for batch_i, (x, y) in enumerate(data_loader):
        x = x.view(x.shape[0], -1).to(device)
        y_onehot = np.zeros((1, 10))
        y_onehot[0, y] = 1
        y_onehot = torch.Tensor(y_onehot).to(device)

        y_score = model(x)
        loss = criterion(y_score, y_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_i % 10000 == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            print(msg.format(epoch+1, batch_i, len(data_loader),
                             batch_i/len(data_loader)*100, loss.item()))



def test(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(data_loader):
            x = x.view(x.shape[0], -1).to(device)
            output = model(x)
            pred = output.cpu().data.max(1)[1]
            correct += pred.eq(y).item()
        print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i + 1,
                                                   100. * correct / (
                                                               batch_i + 1)))


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
    mlp = MLP(x_dim=28*28, y_dim=10, hidden=50)
    mlp.to(device)

    # Dataset of 100 samples (10 per class)
    few_shot_dataset = get_few_shot_mnist(train_loader, shot=10)
    epochs = 30
    for epoch in range(epochs):
        train(epoch, mlp, few_shot_dataset)
        test(mlp, test_loader)

