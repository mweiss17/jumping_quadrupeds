import torch
import numpy as np
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path="data/random-rollouts-200k.zip"):
        self.path = path
        self.data = torch.FloatTensor(torch.moveaxis(torch.load(open(path, "rb")), 3, 1))
        self.data = transforms.Resize(64)(self.data)
        self.data = self.data/255.
        self.data[:, 0] = (self.data[:, 0] - self.data[:, 0].mean())/self.data[:,0].std()
        self.data[:, 1] = (self.data[:, 1] - self.data[:, 1].mean())/self.data[:,1].std()
        self.data[:, 2] = (self.data[:, 2] - self.data[:, 2].mean())/self.data[:,2].std()
        self.data = torch.clip(self.data, -1., 1.)
        self.data = (self.data + 1.0)/2.0

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

  def __getitem__(self, index):
        # return self.transform(self.data[index])
        return self.data[index]
