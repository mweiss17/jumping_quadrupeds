import argparse
import os
import torch
import numpy as np
from torchvision.transforms import transforms


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def common_img_transforms(with_flip=False):
    out = [transforms.Resize(64)]
    if with_flip:
        out.append(transforms.RandomHorizontalFlip())
    out.append(transforms.ToTensor())
    return transforms.Compose(out)


def abs_path(x):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(x)))
