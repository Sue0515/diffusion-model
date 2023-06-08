import torch 
from torch import nn 
from torch.nn import functional as F 


class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__() 
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2), 
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2)
        ])
        self.activation = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for idx, layer in enumerate(self.down_layers):
            x = self.activation(layer(x))
            if idx < 2: # until the final down layer 
                h.append(x) # storing output for skip connection
                x = self.downscale(x)
            
        for idx, layer in enumerate(self.up_layers):
            if idx > 0: # for all except the first up layer 
                x = self.upscale(x)
                x += h.pop() # fetching stored output (skip connection)
            x = self.activation(layer(x))
        
        return x 
    

net = BasicUNet() 
x = torch.rand(8, 1, 28, 28).double() 
net(x.shape)
