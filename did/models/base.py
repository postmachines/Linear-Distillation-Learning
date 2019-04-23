import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim


class RNDModel(nn.Module):
    """
    First version of the model. Model consists of one target isometry
    initialized random nework and N_CLASSES linear predictor networks. As a
    result we have N_CLASSES optimizers each for its predictor.
    """
    def __init__(self, n_classes, dim=784):
        super(RNDModel, self).__init__()

        self.activated_predictor = None
        self.target = nn.Sequential(nn.Linear(dim, dim))
        self.predictors = {}
        self.optimizers = {}
        for c in range(n_classes):
            self.predictors[f'class_{c}'] = nn.Sequential(
                nn.Linear(dim, dim)
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