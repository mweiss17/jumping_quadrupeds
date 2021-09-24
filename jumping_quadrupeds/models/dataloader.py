import torch
import numpy as np
import zipfile
from torchvision import transforms
from PIL import Image, ImageFile
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt


def show_sample(array):
    sampleid = np.random.choice(range(0, len(array)))
    plt.imshow(array[sampleid].moveaxis(0, 2))
    plt.show()

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path="data/random-rollouts-200k.zip", max_num_samples=None):
        self.path = path
        archive = zipfile.ZipFile(path, 'r')
        max_num_samples = max_num_samples if max_num_samples else len(archive.namelist())
        shape = (max_num_samples, 96, 96, 3)
        self.data = torch.zeros(shape)
        for i, f in enumerate(tqdm(archive.namelist(), desc="loading images... wanna multiprocess me?")):
            if i >= max_num_samples:
                break

            img = archive.read(f)
            fh = BytesIO(img)
            img = Image.open(fh)
            self.data[i] = torch.IntTensor(np.asarray(img))
        self.data = torch.FloatTensor(torch.moveaxis(self.data, 3, 1))
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
