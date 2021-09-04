import torch.nn as nn


class LogReg(nn.Module):
    """Basic logistic regression. """

    def __init__(self, n_classes=10, x_dim=28):
        super(LogReg, self).__init__()
        self.model = nn.Linear(x_dim**2, n_classes)

    def forward(self, x):
        out = self.model(x)
        return out