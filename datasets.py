import numpy as np
import torch
import gym
import mnist
import cv2
import sklearn.datasets
import sklearn.model_selection

from cartpole_swingup import CartPoleSwingUpEnv


# Set gym logger to WARN level to stop spamming (only required when logging module is used).
gym.logger.setLevel(40)


def make_env(name):
    """Simple helper function to load CartPoleSwingUp or a gym environment."""
    if name == 'CartPoleSwingUp':
        return CartPoleSwingUpEnv()
    else:
        return gym.make(name)


def load_preprocessed_dataset(name, **kwargs):
    """Simple helper function to load digits or mnist dataset."""
    if name == 'digits':
        return load_preprocessed_digits(**kwargs)
    elif name == 'mnist':
        return load_preprocessed_mnist(**kwargs)
    else:
        raise ValueError('Could not recognize dataset:', name)


def load_preprocessed_digits(use_torch=False, flatten_images=False):
    """
    Return sklearn digits dataset (sklearn.datasets.digits) with images normalized to [0, 1]
    and split into train and test set.

    Args:
        use_torch (bool, optional): If True, return torch tensors, otherwise numpy arrays
            (default: False).
        flatten_images (bool, optional): Flatten train and test images (default: False).

    Returns:
        Train images, train labels, test images, test labels as numpy arrays
        (or torch tensors, see parameter use_torch).
    """
    digits = sklearn.datasets.load_digits()
    normalized_images = digits.images / 16
    train_images, test_images, train_labels, test_labels = sklearn.model_selection.train_test_split(
        normalized_images, digits.target, random_state=0)

    if flatten_images:
        train_images = train_images.reshape(len(train_images), -1)
        test_images = test_images.reshape(len(test_images), -1)

    if use_torch:
        return torch.from_numpy(train_images).float(), torch.from_numpy(train_labels), \
               torch.from_numpy(test_images).float(), torch.from_numpy(test_labels)
    else:
        return train_images, train_labels, test_images, test_labels


def load_preprocessed_mnist(use_torch=False, flatten_images=False):
    """
    Return mnist dataset with images normalized to [0, 1], scaled to (16, 16) and deskewed.

    Args:
        use_torch (bool, optional): If True, return torch tensors, otherwise numpy arrays
            (default: False).
        flatten_images (bool, optional): Flatten train and test images (default: False).

    Returns:
        Train images, train labels, test images, test labels  as numpy arrays
        (or torch tensors, see parameter use_torch).
    """
    train_images = mnist.train_images() / 255
    train_images = preprocess(train_images, (16, 16), unskew=True)
    train_images = train_images.astype('float32')

    test_images = mnist.test_images() / 255
    test_images = preprocess(test_images, (16, 16), unskew=True)
    test_images = test_images.astype('float32')

    train_labels = mnist.train_labels().astype(int)  # original arrays are uint8
    test_labels = mnist.test_labels().astype(int)

    if flatten_images:
        train_images = train_images.reshape(len(train_images), -1)
        test_images = test_images.reshape(len(test_images), -1)

    if use_torch:
        return torch.from_numpy(train_images).float(), torch.from_numpy(train_labels), \
               torch.from_numpy(test_images).float(), torch.from_numpy(test_labels)
    else:
        return train_images, train_labels, test_images, test_labels


def preprocess(img, size, patchCorner=(0, 0), patchDim=None, unskew=True):
    """
    Resize, crop, and unskew images.

    From: https://github.com/google/brain-tokyo-workshop/blob/master/WANNRelease/WANN/domain/classify_gym.py
    """
    if patchDim == None: patchDim = size
    nImg = np.shape(img)[0]
    procImg = np.empty((nImg, size[0], size[1]))

    # Unskew and Resize
    if unskew == True:
        for i in range(nImg):
            procImg[i, :, :] = deskew(cv2.resize(img[i, :, :], size), size)

    # Crop
    cropImg = np.empty((nImg, patchDim[0], patchDim[1]))
    for i in range(nImg):
        cropImg[i, :, :] = procImg[i, patchCorner[0]:patchCorner[0] + patchDim[0], \
                           patchCorner[1]:patchCorner[1] + patchDim[1]]
    procImg = cropImg

    return procImg


def deskew(image, image_shape, negated=True):
    """
    Deskew an image using moments.

    Args:
        image: a numpy nd array input image
        image_shape: a tuple denoting the image`s shape
        negated: a boolean flag telling whether the input image is negated

    Returns:
         a numpy nd array deskewed image

    From: https://github.com/vsvinayak/mnist-helper
    """

    # negate the image
    if not negated:
        image = 255 - image
    # calculate the moments of the image
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    # caclulating the skew
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * image_shape[0] * skew], [0, 1, 0]])
    img = cv2.warpAffine(image, M, image_shape, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
