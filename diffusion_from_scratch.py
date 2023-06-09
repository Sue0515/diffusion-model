import torch 
import torchvision 
from torch import nn 
from torch.nn import functional as F 
from torch.utils.data import DataLoader 
from diffusers import DDPMScheduler, UNet2DModel 
from matplotlib import pyplot as plt 

from unet import BasicUNet

dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

X, y = next(iter(train_dataloader))

def add_noise(img, alpha):
    noise = torch.rand_like(img) # returns a tensor with same size as input that is filled with random numbers
    alpha = alpha.view(-1, 1, 1, 1)
    print(f'Noise: {noise.shape} | alpha: {alpha.shape}')
    noise_added = (1-alpha)*img + noise*alpha

    return noise_added 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters 
batch_size = 128 
epochs = 3 
net = BasicUNet() 
net.to(device)
loss_fn = nn.MSELoss() 
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
losses = []

for epoch in range(epochs):
    for X, y in train_dataloader:
        X = X.to(device)
        noise_alpha = torch.rand(X.shape[0]).to(device)
        noisy_X = add_noise(X, noise_alpha)

        pred = net(noisy_X)
        loss = loss_fn(pred, X)

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        losses.append(loss.item())
    
    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f'Epoch{epoch} - Average Loss: {avg_loss:05f}')

plt.plot(losses)

X, y = next(iter(train_dataloader))
X = X[:8]

alpha = torch.linspace(0, 1, X.shape[0]) # more noise added moving from left batch to right batch 
noised_img = add_noise(X, alpha)

with torch.no_grad():
    pred = net(noised_img.to(device)).detach().cpu() 


fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title('Original Data')
axs[0].imshow(torchvision.utils.make_grid(X)[0].clip(0, 1), cmap='Greys')
axs[1].set_title('Noised Data')
axs[1].imshow(torchvision.utils.make_grid(noised_img)[0].clip(0, 1), cmap='Greys')
axs[2].set_title('Prediction')
axs[2].imshow(torchvision.utils.make_grid(pred)[0].clip(0, 1), cmap='Greys')
plt.show() 
