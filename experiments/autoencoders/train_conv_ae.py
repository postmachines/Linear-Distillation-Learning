import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.utils import save_image

from ldl.data.utils import get_augmented_images
from ldl.data import get_episodic_loader, get_data_loader
from models import ConvAutoEncoder, AEModel


def augment_data(support):
    """
    Augment data by elementary methods.

    Args:
        support (np.ndarray): data of shape [n_way, n_shot, channels, width, hight]

    Returns (np.ndarray): data of shape [n_augmented, width, height]

    """
    w, h = support.shape[-1], support.shape[-1]
    c = support.shape[2]
    x_train = support.squeeze().reshape((-1, c, w, h))
    y_train = [i // train_shot for i in range(train_shot * way)]

    # Shis should be done in preprocessing step
    imgs_aug = []
    y_aug = []
    for i_img in range(x_train.shape[0]):
        img = x_train[i_img].detach().numpy()

        augmented = get_augmented_images(img, shift=4, sigma=0.03)
        imgs_aug += augmented
        y_aug += [y_train[i_img]] * len(augmented)

    x_aug = np.array(imgs_aug, np.float32)
    y_aug = np.array(y_aug)

    return x_aug, y_aug


def train(rnd, loss_func, train_loader, epochs, silent=False):
    rnd.train()
    for epoch in range(epochs):
        for batch_i, (x, y) in enumerate(train_loader):
            #x = x.squeeze().to(device)
            x = x.to(device)
            y = y.to(device)

            # Activate predictor for the needed class
            rnd.activate_predictor(class_=y.item())

            predictor_feature, target_feature = rnd(x)
            predictor_feature = predictor_feature.view(-1, 1)
            target_feature = target_feature.view(-1, 1)
            loss = loss_func(predictor_feature, target_feature).mean()
            optimizer = rnd.get_optimizer(y.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_i % 100 == 0 and not silent:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                print(msg.format(epoch + 1, batch_i, len(train_loader),
                                 batch_i / len(train_loader) * 100, loss.item()))


def test(rnd, test_loader, silent=False):
    rnd.eval()
    correct = 0
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(test_loader):
            x = x.squeeze()
            predict_next_state_feature, target_next_state_feature = rnd.predict(
                x.to(device))
            target_next_state_feature = target_next_state_feature.view(-1, 1)
            mses = []
            for predict in predict_next_state_feature:
                predict = predict.view(-1, 1)
                mses.append(
                    (target_next_state_feature - predict).pow(2).sum(0) / 2)
            class_min_mse = np.argmin(mses)
            if class_min_mse == y.item():
                correct += 1
        acc = correct / (batch_i + 1)
        if not silent:
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, batch_i + 1,
                                                       100. * acc))
    return acc


def train_ae(config, img_dir='did/experiments/ae/conv_ae_images',
             save_path='did/experiments/ae/ae_cnn.pt'):

    # Create directory for images
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Pretrain AE
    w = h = config['x_dim']
    c = config['channels']
    n_epochs = 10
    batch_size = 128
    lr = 0.001

    ae_model = ConvAutoEncoder(n_channels=c)
    ae_model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_model.parameters(),
                                 lr=lr, weight_decay=1e-5)

    dataloader = get_data_loader(dataset=config['dataset'],
                                 split='train',
                                 batch_size=batch_size)
    for epoch in range(n_epochs):
        losses = []
        for x, y in dataloader:
            img = x.cuda()#.unsqueeze(1)

            # Forward
            output = ae_model(img)
            loss = criterion(output, img)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, n_epochs, np.mean(losses)))

        if epoch % 1 == 0:
            pic = to_img(output.cpu().data, c, w, h)
            pic_orig = to_img(img.cpu().data, c, w, h)
            epoch_str = "0"*(4-len(str(epoch))) + str(epoch)
            save_image(pic_orig, f'{img_dir}/image_{epoch_str}_orig.png')
            save_image(pic, f'{img_dir}/image_{epoch_str}.png')

    torch.save(ae_model, save_path)


def to_img(x, c=1, w=28, h=28):
    x = x.view(x.size(0), c, w, h)
    return x


if __name__ == "__main__":
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = {
        'dataset': 'cifar10',
        'way': 10,
        'epochs': 3,
        'train_shot': 10,
        'test_shot': 1,
        'loss': nn.MSELoss(reduction='none'),
        'x_dim': 32,
        'channels': 3,
        'trials': 10,
        'silent': True,
        'split': 'test',
        'add_rotations': True,
        'in_alphabet': True,
        'gpu': 1
    }

    # Pretrain AE
    save_path = 'experiments/autoencoders/ae_cnn.pt'
    #train_ae(config, save_path=save_path)

    ae_model = torch.load(save_path)

    dataset = config['dataset']
    way = config['way']
    train_shot = config['train_shot']
    test_shot = config['test_shot']
    mse_loss = config['loss']
    trials = config['trials']
    w = h = config['x_dim']
    epochs = config['epochs']
    c = config['channels']
    silent = config['silent']
    split = config['split']
    in_alphabet = config['in_alphabet']
    add_rotations = config['add_rotations']

    accs = []
    for _ in tqdm(range(trials)):

        data = get_episodic_loader(dataset, way, train_shot, test_shot,
                                   split=split, add_rotations=add_rotations,
                                   in_alphabet=in_alphabet)

        model = AEModel(way, ae=ae_model, ae_dim=384, w=w, h=h, c=c, cnn=True)
        model.to(device)

        for sample in data:
            support = sample['xs']
            query = sample['xq']

            x_train = support.squeeze().reshape((-1, c, w, h))
            y_train = [i // train_shot for i in range(train_shot * way)]

            #x_train, y_train = augment_data(support)
            x_test = sample['xq']#.reshape((-1, w * h))
            y_test = np.asarray(
                [i // test_shot for i in range(test_shot * way)])

            # break
            x_train = torch.tensor(x_train)
            y_train = torch.tensor(y_train)
            x_test = torch.tensor(x_test)
            y_test = torch.tensor(y_test)

            # print("Train: ", x_train.shape, y_train.shape)
            # print("Test: ", x_test.shape, y_test.shape)

            inds = np.random.permutation(x_train.shape[0])
            samples_train = list(zip(x_train[inds], y_train[inds]))
            samples_test = list(zip(x_test, y_test))

            train(model, loss_func=mse_loss, train_loader=samples_train, epochs=epochs, silent=silent)
            accs.append(test(model, samples_test, silent=silent))

    print("Mean accuracy: ", np.mean(accs))