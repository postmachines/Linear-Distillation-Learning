from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, n_classes=10, x_dim=28, hidden_layers=1, hidden_size=1024,
                 nonlinearity='relu', c=1, table_dataset=False):
        super(MLP, self).__init__()

        self.x_dim = x_dim
        self.channels = c
        
        # Classes for layers
        if nonlinearity == 'relu':
            nnl = nn.ReLU
        elif nonlinearity == 'leaky_relu':
            nnl = nn.PReLU
        
        if table_dataset:
            self.input_size = self.x_dim
        else:
            self.input_size = self.channels * self.x_dim**2
        # Gather layers
        items = [nn.Linear(self.input_size, hidden_size), nnl()]
        for _ in range(hidden_layers-1):
            items.append(nn.Linear(hidden_size, hidden_size))
            items.append(nnl())
        items += [nn.Linear(hidden_size, n_classes)]

        # Aggregate model
        self.model = nn.Sequential(*items)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return F.log_softmax(self.model(x), dim=-1)


class LogReg(nn.Module):

    def __init__(self, n_classes=10, x_dim=28, c=1, table_dataset=False):
        super(LogReg, self).__init__()
        if table_dataset:
            input_size = x_dim
        else:
            input_size = c*x_dim**2
        self.model = nn.Linear(input_size, n_classes)

    def forward(self, x):
        out = self.model(x)
        return out


class TinyCNN(nn.Module):
    def __init__(self, n_classes=10):
        super(TinyCNN, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out