import math
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
from torch.utils.data import RandomSampler, DataLoader, Subset


class Hdf5ImgDataset(Dataset):
    def __init__(self, path, transform=None, flat=False):
        super().__init__()
        self.path = path
        self.flat = flat
        self.transform = transform
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

        if self.flat:
            episode_idx = index % self.episodes
            step_idx = math.floor(index / self.episodes)
            img = self.dataset[episode_idx, step_idx]
        else:
            img = self.dataset[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img


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
