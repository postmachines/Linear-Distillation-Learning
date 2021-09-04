import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init


class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class BidirDistill(nn.Module):
    def __init__(self, n_classes, in_dim, out_dim, lr_predictor=5e-6, lr_target=5e-6, weight_decay=5e-7):
        super(BidirDistill, self).__init__()

        # Current activated predictor
        self.activated_predictor = None

        # Target network
        self.target = Net(in_dim, out_dim)
        self.optimizer_target = optim.Adam(self.target.parameters(),
                                           lr=lr_target,
                                           weight_decay=weight_decay)

        # Predictors
        self.predictors = {}
        self.optimizers = {}
        for c in range(n_classes):
            self.predictors[f'class_{c}'] = Net(in_dim, out_dim)

            self.optimizers[f'class_{c}'] = \
                optim.Adam(self.predictors[f'class_{c}'].parameters(),
                           lr=lr_predictor,
                           weight_decay=weight_decay)

        # Orthogonal initialization
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight)

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
        super(BidirDistill, self).to(device)
        # Move all predictor networks to the same device
        if torch.cuda.is_available():
            for predictor in self.predictors:
                self.predictors[predictor].to(device)