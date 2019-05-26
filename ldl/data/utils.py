import numpy as np
from scipy.ndimage import rotate
from collections import Counter


def get_few_shot_mnist(data_loader, shot=10):
    few_shot_dataset = []
    class_counter = Counter()
    for batch_i, (x, y) in enumerate(data_loader):
        if class_counter[y.item()] < shot:
            class_counter[y.item()] += 1
            few_shot_dataset.append((x, y))
        if all([x == shot for x in class_counter.values()]):
            break
    return few_shot_dataset


def flip(img):
    return np.fliplr(img)


def translate(img, shift=10, direction='right', roll=True):
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img


def gaussian_noise(img, mean=0, sigma=0.04):
    img = img.copy().astype(np.float)
    noise = np.random.normal(mean, sigma, img.shape).astype(np.int16)
    mask_overflow_upper = img+noise >= 1
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 0
    noise[mask_overflow_lower] = 0
    img += noise
    return img


def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    img = rotate(img, angle, reshape=False)
    return img


def get_augmented_images(img, shift=4, sigma=0.03):
    angles = [-30, -20, -10, 0, 10, 20, 30]
    aug_imgs = []

    for angle in angles:
        for trans_dir in ['left', 'right', 'up', 'down']:
            aug_imgs.append(gaussian_noise(translate(rotate_img(img, angle), shift, trans_dir), mean=0, sigma=sigma))
    return aug_imgs
