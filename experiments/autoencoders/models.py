import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_emb(self, x):
        x = x.view(-1, 784)
        return self.encoder(x)


class ConvAutoEncoder(nn.Module):
    def __init__(self, n_channels=1):
        super(ConvAutoEncoder, self).__init__()
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=3, padding=2),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, n_channels, 5, stride=3, padding=2),  # b, 1, 28, 28
            nn.Sigmoid()
        )
        '''
        # For cifar10
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            #nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            # [batch, 3, 32, 32]
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_emb(self, x):
        return self.encoder(x).squeeze()


class AEModel(nn.Module):
    """
    First version of the model. Model consists of one target isometry
    initialized random nework and N_CLASSES linear predictor networks. As a
    result we have N_CLASSES optimizers each for its predictor.
    """
    def __init__(self, n_classes, ae, dim=784, ae_dim=64, w=28, h=28, c=1, cnn=False):
        super(AEModel, self).__init__()

        self.cnn = cnn
        self.activated_predictor = None
        self.ae = ae
        self.target = nn.Sequential(nn.Linear(ae_dim, ae_dim))
        self.predictors = {}
        self.optimizers = {}
        self.w, self.h, self.c = w, h, c
        for c in range(n_classes):
            self.predictors[f'class_{c}'] = nn.Sequential(
                nn.Linear(ae_dim, ae_dim)
            )
            self.optimizers[f'class_{c}'] = \
                optim.Adam(self.predictors[f'class_{c}'].parameters(),
                           0.001)

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight)

        # ae no longer training (but maybe should?)
        for param in self.target.parameters():
            param.requires_grad = False

    def activate_predictor(self, class_):
        self.activated_predictor = self.predictors[f'class_{class_}']

    def get_optimizer(self, class_i):
        return self.optimizers[f"class_{class_i}"]

    def predict(self, next_obs):
        features = self.ae.get_emb(next_obs.view(-1, self.c, self.w, self.h)).view(1, -1)
        predict_features = []
        target_feature = self.target(features)
        for predictor in self.predictors:
            predict_features.append(self.predictors[predictor](features))
        return predict_features, target_feature

    def forward(self, next_obs):
        features = self.ae.get_emb(next_obs.view(-1, self.c, self.w, self.h)).view(1, -1)
        target_feature = self.target(features)
        predict_feature = self.activated_predictor(features)
        return predict_feature, target_feature

    def forward_ae(self, x):
        return self.target(x)

    def to(self, device):
        super(AEModel, self).to(device)
        # Move all predictor networks to the same device
        if torch.cuda.is_available():
            for predictor in self.predictors:
                self.predictors[predictor].to(device)