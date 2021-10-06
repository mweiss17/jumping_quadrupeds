import torch
import os
from tqdm import tqdm
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from jumping_quadrupeds.models.dataloader import Dataset
from jumping_quadrupeds.models.vae import ConvVAE
from speedrun import BaseExperiment, WandBMixin, IOMixin


class TrainLSTM(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):

    def _build(self):

    def run(self):


TrainLSTM().run()
