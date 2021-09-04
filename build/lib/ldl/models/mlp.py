from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, n_classes=10, x_dim=28, hidden_layers=1, hidden_size=1024,
                 nonlinearity='relu'):
        super(MLP, self).__init__()

        self.x_dim = x_dim

        # Classes for layers
        if nonlinearity == 'relu':
            nnl = nn.ReLU
        elif nonlinearity == 'leaky_relu':
            nnl = nn.PReLU

        # Gather layers
        items = [nn.Linear(x_dim**2, hidden_size), nnl()]
        for _ in range(hidden_layers-1):
            items.append(nn.Linear(hidden_size, hidden_size))
            items.append(nnl())
        items += [nn.Linear(hidden_size, n_classes)]

        # Aggregate model
        self.model = nn.Sequential(*items)

    def forward(self, x):
        x = x.view(-1, self.x_dim**2)
        return F.log_softmax(self.model(x), dim=-1)