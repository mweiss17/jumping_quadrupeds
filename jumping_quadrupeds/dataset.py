import h5py
import torch
import numpy as np
import os
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms
import torchvision
from torchvision.datasets import ImageFolder


class Box2dRollout(ImageFolder):
    def __init__(self, *args, **kwargs):
        super(Box2dRollout, self).__init__(*args, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name) for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class Hdf5ImgDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        with h5py.File(path, "r") as f:
            self.episodes = len(f["states"])
            self.steps = len(f["states"][0])
        # IMPORTANT: DO NOT KEEP THE HDF5 OPEN HERE
        # OR ELSE YOU CAN'T USE MULTIPLE DATALOADER WORKERS

    def __len__(self):
        return self.episodes

    def open_ds(self):
        self.file_handle = h5py.File(self.path, "r")
        self.dataset = self.file_handle["states"]

    def __getitem__(self, index):
        if not hasattr(self, "file_handle"):
            self.open_ds()
        return self.dataset[index]


class Hdf5ImgSeqDataset(Hdf5ImgDataset):
    def __init__(self, path, transform=None):
        super().__init__(path)
        self.transform = transform

    def __getitem__(self, index):
        if not hasattr(self, "file_handle"):
            self.open_ds()
        imgs = self.dataset[index]

        out = []
        for img in imgs:
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            out.append(img.unsqueeze(0))
        out = torch.cat(out)
        return out


class MySubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class ClipAndRescale:
    def __init__(self, _min, _max):
        self.min = _min
        self.max = _max

    def __call__(self, img):
        img = torch.clamp(img, self.min, self.max)
        img = (img + 1) / 2
        return img
