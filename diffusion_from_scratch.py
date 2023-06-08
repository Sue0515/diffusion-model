import torch 
import torchvision 
from torch import nn 
from torch.nn import functional as F 
from torch.utils.data import DataLoader 
from diffusers import DDPMScheduler, UNet2DModel 
from matplotlib import pyplot as plt 

dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

X, y = next(iter(train_dataloader))
plt.imshow(torchvision.utils.make_grid(X)[0], cmap='Greys')

def add_noise(img, alpha):
    noise = torch.rand_like(img) # returns a tensor with same size as input that is filled with random numbers
    alpha = alpha.view(-1, 1, 1, 1)
    # print(f'Noise: {noise.shape} | alpha: {alpha.shape}')
    noise_added = (1-alpha)*img + noise*alpha

    return noise_added 


alpha = torch.linspace(0, 1, X.shape[0]) # more noise added moving from left batch to right batch 
noised_img = add_noise(X, alpha)

fig, axs = plt.subplots(2, 1, figsize=(12, 5))
axs[0].set_title('Input')
axs[0].imshow(torchvision.utils.make_grid(X)[0], cmap='Greys')
axs[1].set_title('Corrupted')
axs[1].imshow(torchvision.utils.make_grid(noised_img)[0], cmap='Greys')
plt.show() # When you are plotting a graph in a script, make sure to use the following command to output the window displaying the graph