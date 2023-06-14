import numpy as np
import torch
import torchvision
import open_clip
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DDIMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

device = ('cuda' if torch.cuda.is_available() else 'cpu')

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model.to(device)

def hue_loss(img, tgt_hue):
    target = (torch.tensor(tgt_hue)).to(img.device) * 2 -1 # map to (-1, 1)
    target = target[None, :, None, None]
    err = torch.abs(img-target).mean() # mean absolute error 

    return err 

def clip_loss(img, text):
    img_feat = clip_model.encode_image(transformations(img))
    input_normed = torch.nn.functional.normalize(img_feat.unsqueeze(1), dim=2)
    embed_normed = torch.nn.functional.normalize(text.unsqueeze(0), dim=2)
    sgcd = (input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2))  # Squared Great Circle Distance
    
    return sgcd.mean() 
 
pipeline_name = "johnowhitaker/sd-class-wikiart-from-bedrooms"

image_pipe = DDPMPipeline.from_pretrained(pipeline_name).to(device)
scheduler = DDIMScheduler.from_pretrained(pipeline_name)
scheduler.set_timesteps(num_inference_steps=40)
guidance_loss_scale = 100 # determines the strength of the effect 
noise = torch.randn(8, 3, 256, 256).to(device)

for idx, t in tqdm(enumerate(scheduler.timesteps)):
    input = scheduler.scale_model_input(noise, t)

    with torch.no_grad(): 
        noise_pred = image_pipe.unet(input, t)['sample'] # predict without grad, predict noise residual 
    
    noise = noise.detach().requires_grad_() 
    img_0 = scheduler.step(noise_pred, t, noise).pred_original_sample 
    loss = hue_loss(img_0)*guidance_loss_scale 
    
    grad = -torch.autograd.grad(loss, noise)[0]

    noise = noise.detach() + grad 
    noise = scheduler.step(noise_pred, t, noise).prev_sample 

grid = torchvision.utils.make_grid(noise, nrow=4)
image = grid.permute(1, 2, 0).cpu().clip(-1, 1)*0.5+0.5
Image.fromarray(np.array(image*255).astype(np.uint8))

####### CLIP Guidance 
prompt = 'A Painting of Purple Tulip'
text = open_clip.tokenize([prompt]).to(device)
guidance_loss_scale = 8 
n_cuts = 4 
scheduler.set_timesteps(50) # more steps, more time for the guidance to be effective 

with torch.no_grad(), torch.cuda.amp.autocast():
    text_feat = clip_model.encode_text(text)

noise = torch.randn(4, 3 ,256, 256).to(device)

for idx, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(noise, t)

    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)['sample']
    
    cond_grad = 0 

    for cut in (n_cuts): # compare between hue guidance 
        noise = noise.detach().requires_grad_() 
        img_0 = scheduler.step(noise_pred, t, noise).pred_original_sample 
        loss = clip_loss(img_0, text_feat)*guidance_loss_scale
        cond_grad -= torch.autograd.grad(loss, noise)[0] / n_cuts 
    
    if idx % 10 == 0:
        print(f'Step: {idx} | Guidance Loss: {loss.item()}')
    
    alpha_bar = scheduler.alphas_cumprod[idx]
    noise = noise.detach() + cond_grad*alpha_bar.sqrt() 
    noise = scheduler.step(noise_pred, t, noise).prev_sample 

grid = torchvision.utils.make_grid(noise.detach(), nrow=4)
image = grid.permute(1, 2, 0).cpu().clip(-1, 1)*0.5+0.5
Image.fromarray(np.array(image*255).astype(np.uint8))


