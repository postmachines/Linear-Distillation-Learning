import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim


class DLDReversed(nn.Module):
    """
    First version of the model. Model consists of one target isometry
    initialized random nework and N_CLASSES linear predictor networks. As a
    result we have N_CLASSES optimizers each for its predictor.
    """
    def __init__(self, n_classes, in_dim=784, out_dim=784, opt='adam', lr=0.001,
                 initialization='orthogonal'):
        super(DLDReversed, self).__init__()

        self.activated_predictor = None
        self.target = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.predictors = {}
        self.optimizers = {}

        opt = opt.lower()
        if opt == 'adam':
            opt = optim.Adam
        elif opt == 'adadelta':
            opt = optim.Adadelta
        elif opt == 'adagrad':
            opt = optim.Adagrad
        elif opt == 'sgd':
            opt = optim.SGD
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        for c in range(n_classes):
            self.predictors[f'class_{c}'] = nn.Sequential(
                nn.Linear(in_dim, out_dim)
            )
            self.optimizers[f'class_{c}'] = \
                opt(self.predictors[f'class_{c}'].parameters(), lr)

        for p in self.modules():
            if isinstance(p, nn.Linear):
                if initialization == 'orthogonal':
                    init.orthogonal_(p.weight)
                elif initialization == 'xavier_normal':
                    init.xavier_normal_(p.weight)
                elif initialization == 'xavier_uniform':
                    init.xavier_uniform_(p.weight)
                else:
                    raise ValueError(f"Unknown initialization function: {initialization}")

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
            predict_features.append(self.predictors[predictor](target_feature))
        return predict_features

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.activated_predictor(target_feature)
        return predict_feature

    def to(self, device):
        super(DLDReversed, self).to(device)
        # Move all predictor networks to the same device
        if torch.cuda.is_available():
            for predictor in self.predictors:
                self.predictors[predictor].to(device)