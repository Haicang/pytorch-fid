"""
Based on https://github.com/sbarratt/inception-score-pytorch
"""

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

from PIL import Image
import numpy as np
from scipy.stats import entropy

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

__all__ = [
    'NCEPTION_DEFAULT_IMAGE_SIZE',
    'inception_score'
]

INCEPTION_DEFAULT_IMAGE_SIZE = 299
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=1,
                    help=('Path to the generated images'))

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(tqdm(dataloader), 0):
        batch = batch.type(dtype)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def is_image_file(filename):
    return (not filename.startswith('.')) and (filename.endswith(IMG_EXTENSIONS))


class ResultDataset(torch.utils.data.Dataset):
    """
    Assume all generated images in path, with extension `.png`
    """
    def __init__(self, path, transform=None, loader=dset.folder.default_loader):
        self.path = path
        self.transform = transform
        self.files = list(filter(is_image_file , os.listdir(path)))
        self.loader = loader

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = os.path.join(self.path, self.files[index])
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    args = parser.parse_args()

    test_data = ResultDataset(
        args.path[0], transform=transforms.Compose([
            transforms.Resize(INCEPTION_DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    print("Calculating Inception Score... %d images in total" % (len(test_data)))
    print(inception_score(test_data, cuda=True, batch_size=16))
