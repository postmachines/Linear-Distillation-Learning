import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim


class RNDLogModel(nn.Module):
    """
    First version of the model. Model consists of one target isometry
    initialized random nework and N_CLASSES linear predictor networks. As a
    result we have N_CLASSES optimizers each for its predictor.
    """
    def __init__(self, n_classes, in_dim=784, out_dim=2000, opt='adam', lr=0.001,
                 lr_target=0.001, initialization='orthogonal', dld=False):
        super(RNDLogModel, self).__init__()

        self.activated_predictor = None
        self.target_half_1 = nn.Linear(in_dim, out_dim)
        self.target_half_2 = nn.Linear(out_dim, n_classes)
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
        
        target_params = list(self.target_half_1.parameters()) + list(self.target_half_2.parameters())
        
        self.optimizer_target = opt(target_params,
                                   lr=lr_target)

            
        for c in range(n_classes):
            if dld:
                self.predictors[f'class_{c}'] = nn.Sequential(
                    nn.Linear(in_dim, 2000),
                    nn.Linear(2000, out_dim)
                )
            else:
                self.predictors[f'class_{c}'] = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
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


    def activate_predictor(self, class_):
        self.activated_predictor = self.predictors[f'class_{class_}']

    def get_optimizer(self, class_i):
        return self.optimizers[f"class_{class_i}"]
    
    def get_optimizer_target(self):
        return self.optimizer_target

    def predict(self, next_obs):
        predict_features = []
        target_feature = self.target_half_1(next_obs)
        for predictor in self.predictors:
            predict_features.append(self.predictors[predictor](next_obs))
        return predict_features, target_feature

    def forward(self, next_obs, target_path='full'):
        if target_path == 'full':
            target_feature = self.target_half_1(next_obs)
            target_feature = self.target_half_2(target_feature)
            return target_feature
        
        target_feature = self.target_half_1(next_obs)
        predict_feature = self.activated_predictor(next_obs)
        return predict_feature, target_feature

    def to(self, device):
        super(RNDLogModel, self).to(device)
        # Move all predictor networks to the same device
        if torch.cuda.is_available():
            for predictor in self.predictors:
                self.predictors[predictor].to(device)