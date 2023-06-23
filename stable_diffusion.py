from base64 import b64encode

import numpy
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login

from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging

torch.manual_seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae')
tokenizer = CLIPTokenizer.from_pretrained('openai/cli-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet')
vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae')
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)



vae = vae.to(device)
text_encoder = text_encoder.to(device)
unet = unet.to(device)

prompt = ['An oil painting of a rabbit']
h, w = 512, 512
inference_steps = 30 # num of denoising steps 
guidance_scale = 7.5 # scale for classifier-free guidance 
generator = torch.manual_seed(32)
batch_size = 1

# Prep text 
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(['']*batch_size, padding='max_length', max_length=max_length, return_tensoors='pt')
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
text_embedings = torch.cat([uncond_embeddings, text_embeddings])

scheduler.set_timesteps(inference_steps)

latents = torch.randn(batch_size, unet.in_channels, h//8, w//8)
latents = latents.to(device)
latents = latents* scheduler.init_noise_sigma # scaling 

# with autocast('cuda'):



