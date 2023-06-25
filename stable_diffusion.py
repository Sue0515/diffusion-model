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

# Autoencoder 
def img_to_latent(img):
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(img).unsqueeze(0).to(device)*2-1) 
    return 0.18215 * latent.latent_dist.sample()

def latent_to_img(latents):
    latents = (1 / 0.18215) * latents

    with torch.no_grad():
        img = vae.decode(latents).sample

    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (img * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in imgs]
    return pil_images


torch.manual_seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_img = Image.open('puppy.jpg').resize((512, 512))

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

# latents = torch.randn(batch_size, unet.in_channels, h//8, w//8)
# latents = latents.to(device)
# latents = latents* scheduler.init_noise_sigma # scaling 
encoded_img = img_to_latent(input_img)
start_sigma = scheduler.sigmas[10]
noise = torch.randn_like(encoded_img)
latents = scheduler.add_noise(encoded_img, noise, timesteps=torch.tensor([scheduler.timesteps[10]]))
latents = latents.to(device).float() 

# sampling loop
for i, t in tqdm(enumerate(scheduler.timesteps)):
    if i >= 10:
        latent_model_input = torch.cat([latents*2])
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        sigma = scheduler.sigmas[i]

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t ,encoder_hidden_states=text_embeddings)['sample']
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale *(noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample 

latent_to_img(latents)[0]