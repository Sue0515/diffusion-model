import torch
from torch import nn
from torch.nn import functional as F
from diffusers import UNet2DModel

class ConditionedUNet(nn.Module):
    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__() 
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.net = UNet2DModel(
            sample_size=28,
            in_channels=1 + class_emb_size, # why is the class_emb_size 4? 
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 64),
            down_block_types=(
                'DownBlock2D',
                'AttnDownBlock2D',
                'AttnDownBlock2D'
            ),
            up_block_types=(
                'AttnUpBlock2D',
                'AttnUpBlock2D',
                'UpBlock2D'
            )
        )
    
    def forward(self, x, t, labels):
        b, c, w, h = x.shape 
        class_cond = self.class_emb(labels) # shape of class_cond? 
        class_cond = class_cond.view(b, class_cond.shape[1], 1, 1).expand(b, class_cond.shape[1], w, h) # x: (b, 1, 28, 28) | class_cond: (b, 4, 28, 28)
       
        net_input = torch.cat((x, class_cond), 1) # net input: x and class_cond concatenated along axis 1 : (bs, 5, 28, 28)
        res = self.model(net_input, t).sample # Feed this to the unet alongside the timestep 
        return res # (b, 1, 28, 28)