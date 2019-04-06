import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.optim as optim
import torchvision
from collections import Counter

from utils import global_grad_norm_

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def init_weights(m):
    if type(m)==nn.Linear:
        init.orthogonal_(m.weight)


class RNDModel(nn.Module):
    def __init__(self, n_classes):
        super(RNDModel, self).__init__()

        self.activated_predictor = None
        self.target = nn.Sequential(nn.Linear(784, 784))
        self.predictors = {}
        for c in range(n_classes):
            self.predictors[f'predictor_{c}'] = nn.Sequential(
                nn.Linear(784, 784),
            )

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight)

        for param in self.target.parameters():
            param.requires_grad = False
        for predictor in self.predictors:
            for param in self.predictors[predictor].parameters():
                param.requires_grad = False

    def cuda_predictors(self):
        if torch.cuda.is_available():
            for predictor in self.predictors:
                self.predictors[predictor].cuda()

    def activate_predictor(self, class_):
        self.activated_predictor = self.predictors[f'predictor_{class_}']
        for param in self.activated_predictor.parameters():
            param.requires_grad = True

    def deactivate_predictor(self):
        for param in self.activated_predictor.parameters():
            param.requires_grad = False

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


# Batch size must be 1!
def train(epoch, rnd, train_loader, shots_num):
    for batch_i, (x, y) in enumerate(train_loader):
        x = x.view(x.shape[0], -1)
        rnd.activate_predictor(class_=y.cpu().numpy()[0])

        predict_next_state_feature, target_next_state_feature = rnd(
            Variable(x.to(device)))
        forward_loss = forward_mse(predict_next_state_feature,
                                   target_next_state_feature.detach()).mean(-1)
        forward_loss = forward_loss.sum() / len(forward_loss)

        optimizer = optim.Adam(list(rnd.activated_predictor.parameters()),
                               lr=0.001)
        optimizer.zero_grad()
        loss = forward_loss
        loss.backward()
        global_grad_norm_(list(rnd.activated_predictor.parameters()))
        optimizer.step()

        if batch_i % 1000 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_i * len(
                                                                             x),
                                                                         shots_num,
                                                                         100. * batch_i / shots_num,
                                                                         loss.item()))


def test(rnd, test_loader):
    rnd.eval()
    correct = 0
    mses = []
    with torch.no_grad():
        for batch_i, (data, y)  in enumerate(test_loader):
            data = data.view(data.shape[0],-1 )
            predict_next_state_feature, target_next_state_feature = rnd.predict(Variable(data.to(device)))
            for predict in predict_next_state_feature:
                mses.append((target_next_state_feature - predict).pow(2).sum(1) / 2)
            min_mse = np.argmin(mses)
            if min_mse==y.cpu().numpy()[0]:
                correct+=1
            mses = []
        print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i+1, 100. * correct / (batch_i+1)))


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/MNIST/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=1, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/MNIST/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=1, shuffle=True)

    rnd = RNDModel(10)
    rnd.to(device)
    rnd.cuda_predictors()

    params = []
    for _, predictor in rnd.predictors.items():
        params += list(predictor.parameters())

    optimizer = optim.Adam(params, lr=0.001)
    forward_mse = nn.MSELoss(reduction='none')

    update_proportion = 0.25

    num_of_shots = 11
    break_treshold = num_of_shots * 20
    few_shot_dataset = []
    few_shot_dataset_y = []
    few_shot_dataset_y_np = list(range(0, 10))
    for batch_idx, (data, target) in enumerate(train_loader):
        num_of_samples = [x for x in Counter(few_shot_dataset_y_np).values()]
        pos_of_samples = [x for x in Counter(few_shot_dataset_y_np).keys()]
        if num_of_samples[
            pos_of_samples.index(target.cpu().numpy()[0])] < num_of_shots:
            few_shot_dataset.append(data)
            few_shot_dataset_y.append(target)
            few_shot_dataset_y_np.append(target.cpu().numpy()[0])
        if batch_idx > break_treshold:
            break

    test(rnd, test_loader)

    for epoch in range(1, 2):
        train(epoch, rnd, zip(few_shot_dataset, few_shot_dataset_y),
              len(few_shot_dataset))
        test(rnd, test_loader)
