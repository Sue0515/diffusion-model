import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from conditioned_unet import ConditionedUNet

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=False, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

img, label = next(iter(train_dataloader))

net = ConditionedUNet().to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

n_epochs = 10 
loss_fn = nn.MSELoss() 
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

losses = []

for epoch in range(n_epochs):
    for img, label in tqdm(train_dataloader):
        img = img.to(device)*2-1 # Data on the GPU (mapped to (-1, 1))
        label = label.to(device)
        noise = torch.randn_like(X)
        timesteps = torch.randint(0, 999, (img.shape[0],)).long().to(device)
        noisy_img = noise_scheduler.add_noise(img, noise, timesteps)

        prediction = net(noisy_img, timesteps, label)
        loss = loss_fn(prediction, noise)

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        losses.append(loss.item())

    avg_loss = sum(losses[-100:])/100 
    print(f'Epoch{epoch} - Average Loss: {avg_loss:05f}')

# View the loss curve
plt.plot(losses)

X = torch.randn(80, 1, 28, 28).to(device)
y = torch.tensor([i]*8 for i in range(10)).flatten().to(device)

for idx, t in tqdm(enumerate(noise_scheduler.timesteps)):
    with torch.no_grad():
        residual = net(X, t, y)

    X = noise_scheduler.step(residual, t, X).prev_sample # update the sample with step 

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(torchvision.utils.make_grid(X.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')
     

 