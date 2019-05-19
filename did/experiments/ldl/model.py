import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 2000, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class LDL(nn.Module):
    def __init__(self, n_classes, lr=5e-6, tar_lr=5e-6, dim=784):
        super(LDL, self).__init__()

        self.predictors = {}
        self.optimizers = {}
        self.activated_predictor = None

        self.target = Net()

        self.optimizer_target = \
            optim.Adam(self.target.parameters(), tar_lr, weight_decay=5e-7)

        for c in range(n_classes):
            self.predictors[f'class_{c}'] = Net()

            self.optimizers[f'class_{c}'] = \
                optim.Adam(self.predictors[f'class_{c}'].parameters(), lr,
                           weight_decay=5e-7)

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight)

    def reinitialize_predictors(self):
        for p in self.predictors:
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
        super(LDL, self).to(device)
        # Move all predictor networks to the same device
        if torch.cuda.is_available():
            for predictor in self.predictors:
                self.predictors[predictor].to(device)