import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class SIMMResNet18(nn.Module):
    """
    First version of the model. Model consists of one target isometry
    initialized random nework and N_CLASSES linear predictor networks. As a
    result we have N_CLASSES optimizers each for its predictor.
    """
    def __init__(self, n_classes, dim=784):
        super(SIMMResNet18, self).__init__()

        self.activated_predictor = None

        # Pretrained CNN
        num_ftrs = 512
        resnet = torchvision.models.resnet18(pretrained=True)
        self.target = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictors = {}
        self.optimizers = {}
        for c in range(n_classes):
            self.predictors[f'class_{c}'] = nn.Sequential(
                nn.Linear(dim, num_ftrs)
            )
            #for p in self.predictors[f'class_{c}'].modules():
            #    init.orthogonal_(p)
            self.optimizers[f'class_{c}'] = \
                optim.Adam(self.predictors[f'class_{c}'].parameters(), 0.001)

    def activate_predictor(self, class_):
        self.activated_predictor = self.predictors[f'class_{class_}']

    def get_optimizer(self, class_i):
        return self.optimizers[f"class_{class_i}"]

    def predict(self, next_obs):
        predict_features = []
        x_target = next_obs.unsqueeze_(0).repeat(2,3,1,1)
        target_feature = self.target(x_target)[0,:]
        x_predictor = next_obs.view((-1, 28 * 28))
        for predictor in self.predictors:
            predict_features.append(self.predictors[predictor](x_predictor).squeeze_())

        return predict_features, target_feature.squeeze_()

    def forward(self, next_obs):
        x_target = next_obs.unsqueeze_(0).repeat(2,3,1,1)
        target_feature = self.target(x_target)[0,:]
        x_predictor = next_obs.view((-1, 28*28))
        predict_feature = self.activated_predictor(x_predictor)

        return predict_feature.squeeze_(), target_feature.squeeze_()

    def to(self, device):
        super(SIMMResNet18, self).to(device)
        # Move all predictor networks to the same device
        if torch.cuda.is_available():
            for predictor in self.predictors:
                self.predictors[predictor].to(device)
        self.target.to(device)