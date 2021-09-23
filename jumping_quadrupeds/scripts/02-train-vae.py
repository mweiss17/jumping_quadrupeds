import torch
import os
from tqdm import tqdm
import numpy as np
import scipy.misc
from torch import optim
import matplotlib.pyplot as plt
import jumping_quadrupeds
from jumping_quadrupeds.models.dataloader import Dataset
from jumping_quadrupeds.models.vae import ConvVAE

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100
print(os.getcwd())

# Generators
dataset = Dataset()

split = dataset.data.shape[0]//8
train = torch.utils.data.DataLoader(dataset[split:], **params)
valid = torch.utils.data.DataLoader(dataset[:split], **params)

model = ConvVAE(in_channels=3, latent_dim=96)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.95)
# Loop over epochs
for epoch in tqdm(range(max_epochs), desc="epochs..."):
    for batch in train:
        batch = batch.to(device)
        x_hat, input, mu, log_var = model(batch)
        loss = model.loss_function(x_hat, input, mu, log_var, M_N=1)
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
        print(f"loss: {loss}, lr: {scheduler.get_lr()}")
    scheduler.step()
    if not os.path.isdir("vae_results"):
        os.makedirs("vae_results")
    true_image = batch[0].detach().transpose(0, 2).cpu().numpy()
    plt.imsave(f"vae_results/true-epoch{epoch}.jpg", true_image)
    generated = x_hat[0].detach().transpose(0, 2).cpu().numpy()
    generated = ((generated+1)/2)
    plt.imsave(f"vae_results/generated-epoch{epoch}.jpg", generated)

    # # Validation
    # with torch.set_grad_enabled(False):
    #     for local_batch, local_labels in validation_generator:

