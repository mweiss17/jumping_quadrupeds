import torch
import numpy as np
import os
from typing import Dict, List, Tuple
from torchvision import transforms
import torchvision

class Box2dRollout(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(Box2dRollout, self).__init__(*args, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name) for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


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


class ClipAndRescale():
    def __init__(self, _min, _max):
        self.min = _min
        self.max = _max

    def __call__(self, img):         
        img = torch.clamp(img, self.min, self.max)
        img = (img + 1) / 2
        return img