###################################################################################################################################
#
# Development of a Command-Line Interface for High-Resolution Image Generation using Stable Diffusion and Upsampling Techniques
#
###################################################################################################################################

import sys, os
sys.path.extend(['./taming-transformers', './stable-diffusion', './latent-diffusion'])

import numpy as np
import time
import re
import requests
import hashlib
from subprocess import Popen
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm, trange
from functools import partial
from ldm.util import instantiate_from_config
import k_diffusion as K
from requests.exceptions import HTTPError
import huggingface_hub
import argparse

from upscaler_utils.CFGUpscaler import CFGUpscaler
from upscaler_utils.NoiseLevelAndTextConditionedUpscaler import NoiseLevelAndTextConditionedUpscaler

from clip_utils.CLIPEmbedder import CLIPEmbedder
from clip_utils.CLIPTokenizerTransform import CLIPTokenizerTransform
from generate import SDImageGenerator

from dotenv import load_dotenv
load_dotenv()

# Model configuration values
SD_C = 4 # Latent dimension
SD_F = 8 # Latent patch size (pixels per latent)
SD_Q = 0.18215 # sd_model.scale_factor; scaling for latents in first stage models

def clean_prompt(prompt):
  badchars = re.compile(r'[/\\]')
  prompt = badchars.sub('_', prompt)
  if len(prompt) > 100:
    prompt = prompt[:100] + '…'
  return prompt

def format_filename(timestamp, seed, index, prompt):
  string = save_location
  string = string.replace('%T', f'{timestamp}')
  string = string.replace('%S', f'{seed}')
  string = string.replace('%I', f'{index:02}')
  string = string.replace('%P', clean_prompt(prompt))
  return string

def save_image(image, **kwargs):
  filename = format_filename(**kwargs)
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  image.save(filename)

def fetch(url_or_path):
  if url_or_path.startswith('http:') or url_or_path.startswith('https:'):
    _, ext = os.path.splitext(os.path.basename(url_or_path))
    cachekey = hashlib.md5(url_or_path.encode('utf-8')).hexdigest()
    cachename = f'{cachekey}{ext}'
    if not os.path.exists(f'cache/{cachename}'):
      os.makedirs('tmp', exist_ok=True)
      os.makedirs('cache', exist_ok=True)

      response = requests.get(url_or_path)
      with open(f'tmp/{cachename}', 'wb') as f:
            f.write(response.content)
      os.rename(f'tmp/{cachename}', f'cache/{cachename}')
    return f'cache/{cachename}'
  return url_or_path

def make_upscaler_model(config_path, model_path, pooler_dim=768, train=False, device='cpu'):
    config = K.config.load_config(open(config_path))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config['model']['sigma_data'],
        embed_dim=config['model']['mapping_cond_dim'] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_ema'])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)

def download_from_huggingface(repo, filename):
  while True:
    try:
      return huggingface_hub.hf_hub_download(repo, filename)
    except HTTPError as e:
      if e.response.status_code == 401:
        # Need to log into huggingface api
        huggingface_hub.interpreter_login()
        continue
      elif e.response.status_code == 403:
        # Need to do the click through license thing
        print(f'Go here and agree to the click through license on your account: https://huggingface.co/{repo}')
        input('Hit enter when ready:')
        continue
      else:
        raise e

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model = model.to(cpu).eval().requires_grad_(False)
    return model

@torch.no_grad()
def condition_up(prompts):
  return text_encoder_up(tok_up(prompts))

@torch.no_grad()
def run(seed):
  timestamp = int(time.time())
  if not seed:
    print('No seed was provided, using the current time.')
    seed = timestamp
  print(f'Generating with seed={seed}')
  seed_everything(seed)


  uc = condition_up(batch_size * [""])
  c = condition_up(batch_size * [prompt])

  if decoder == 'finetuned_840k':
    vae = vae_model_840k
  elif decoder == 'finetuned_560k':
    vae = vae_model_560k

  # image = Image.open(fetch(input_file)).convert('RGB')
  image = input_image
  image = TF.to_tensor(image).to(device) * 2 - 1
  low_res_latent = vae.encode(image.unsqueeze(0)).sample() * SD_Q
  low_res_decoded = vae.decode(low_res_latent/SD_Q)

  [_, C, H, W] = low_res_latent.shape

  # Noise levels from stable diffusion.
  sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

  model_wrap = CFGUpscaler(model_up, uc, cond_scale=guidance_scale)
  low_res_sigma = torch.full([batch_size], noise_aug_level, device=device)
  x_shape = [batch_size, C, 2*H, 2*W]

  def do_sample(noise, extra_args):
    # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
    sigmas = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps+1).exp().to(device)
    if sampler == 'k_euler':
      return K.sampling.sample_euler(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args)
    elif sampler == 'k_euler_ancestral':
      return K.sampling.sample_euler_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
    elif sampler == 'k_dpm_2_ancestral':
      return K.sampling.sample_dpm_2_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
    elif sampler == 'k_dpm_fast':
      return K.sampling.sample_dpm_fast(model_wrap, noise * sigma_max, sigma_min, sigma_max, steps, extra_args=extra_args, eta=eta)
    elif sampler == 'k_dpm_adaptive':
      sampler_opts = dict(s_noise=1., rtol=tol_scale * 0.05, atol=tol_scale / 127.5, pcoeff=0.2, icoeff=0.4, dcoeff=0)
      return K.sampling.sample_dpm_adaptive(model_wrap, noise * sigma_max, sigma_min, sigma_max, extra_args=extra_args, eta=eta, **sampler_opts)

  image_id = 0
  for _ in range((num_samples-1)//batch_size + 1):
    if noise_aug_type == 'gaussian':
      latent_noised = low_res_latent + noise_aug_level * torch.randn_like(low_res_latent)
    elif noise_aug_type == 'fake':
      latent_noised = low_res_latent * (noise_aug_level ** 2 + 1)**0.5
    extra_args = {'low_res': latent_noised, 'low_res_sigma': low_res_sigma, 'c': c}
    noise = torch.randn(x_shape, device=device)
    up_latents = do_sample(noise, extra_args)

    pixels = vae.decode(up_latents/SD_Q) # equivalent to sd_model.decode_first_stage(up_latents)
    pixels = pixels.add(1).div(2).clamp(0,1)


    # Display and save samples.
    # display(TF.to_pil_image(make_grid(pixels, batch_size)))
    for j in range(pixels.shape[0]):
      img = TF.to_pil_image(pixels[j])
      save_image(img, timestamp=timestamp, index=image_id, prompt=prompt, seed=seed)
      image_id += 1

##########################
# 0.Command line Argument 
##########################

parser = argparse.ArgumentParser(prog='PROG', usage='main.py [-h] [--prompt HOGE] [--seed 1010100]', description='DESCRIPTION', epilog='EPILOG')
parser.add_argument('-p','--prompt', type=str, help='input prompt (string type)')
parser.add_argument('-s','--seed', type=int, help='setting seed (integer type)')

args = parser.parse_args()
prompt = args.prompt
seed = args.seed

engine_id = os.getenv('SD_ENGINE_ID')
generator_key = os.getenv('SD_API_SECRET_KEY')
engine_id = "stable-diffusion-v1-5"
generator = SDImageGenerator(engine_id, generator_key)

##########################
# 1.Prepare
##########################

# Save location format:
# %T: timestamp
# %S: seed
# %I: image index
# %P: prompt (will be truncated to avoid overly long filenames)
SAVE_DIR = "./outputs/"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Directory '{SAVE_DIR}' has been created.")
else:
    print(f"Directory '{SAVE_DIR}' already exists. Skipping creation.")

resolution_dir = "1024_1024/"
generated_dir = os.path.join(SAVE_DIR, resolution_dir)
if not os.path.exists(generated_dir):
    os.makedirs(generated_dir)
    print(f"Directory '{generated_dir}' has been created.")
else:
    print(f"Directory '{generated_dir}' already exists. Skipping creation.")

save_location = generated_dir +'%T-%I-%P.png' 

##########################
# 1.1 Fetch models
##########################

model_up = make_upscaler_model(fetch('https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json'),
                               fetch('https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth'))

# sd_model_path = download_from_huggingface("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
vae_840k_model_path = download_from_huggingface("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.ckpt")
vae_560k_model_path = download_from_huggingface("stabilityai/sd-vae-ft-ema-original", "vae-ft-ema-560000-ema-pruned.ckpt")


##########################
# 1.2 Load models on GPU
##########################

cpu = torch.device("cpu")
# device = torch.device("cuda")
device = torch.device("cpu")

# sd_model = load_model_from_config("stable-diffusion/configs/stable-diffusion/v1-inference.yaml", sd_model_path)
vae_model_840k = load_model_from_config("latent-diffusion/models/first_stage_models/kl-f8/config.yaml", vae_840k_model_path)
vae_model_560k = load_model_from_config("latent-diffusion/models/first_stage_models/kl-f8/config.yaml", vae_560k_model_path)

# sd_model = sd_model.to(device)
vae_model_840k = vae_model_840k.to(device)
vae_model_560k = vae_model_560k.to(device)
model_up = model_up.to(device)


######################################################
# 1.3 Set up some functions and load the text encoder
######################################################

tok_up = CLIPTokenizerTransform()
text_encoder_up = CLIPEmbedder(device=device)

##########################
# 2.Configuration and Run
##########################

##########################
# 2.1 Configuration
##########################

# Prompt. Not strictly required but can subtly affect the upscaling result.
num_samples = 1 
batch_size = 1 

decoder = 'finetuned_840k' # ["finetuned_840k", "finetuned_560k"]

guidance_scale = 1 # min: 0.0, max: 10.0

# Add noise to the latent vectors before upscaling. This theoretically can make the model work better on out-of-distribution inputs, but mostly just seems to make it match the input less, so it's turned off by default.
noise_aug_level = 0 # min: 0.0, max: 0.6
noise_aug_type = 'gaussian' # ["gaussian", "fake"]


# Sampler settings. `k_dpm_adaptive` uses an adaptive solver with error tolerance `tol_scale`, all other use a fixed number of steps.
sampler = 'k_dpm_adaptive' # ["k_euler", "k_euler_ancestral", "k_dpm_2_ancestral", "k_dpm_fast", "k_dpm_adaptive"]
steps = 50 
tol_scale = 0.25 
# Amount of noise to add per step (0.0=deterministic). Used in all samplers except `k_euler`.
eta = 1.0

if not prompt:
  print('no prompt! Initialize...')
  prompt = "the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas"

print("inputed prompt:", prompt)

############################################
# 2.2 Generate Image from Text (512×512)
############################################

# This works best on images generated by stable diffusion.
if 'input_image' not in globals():
  input_image = generator.generate_image(prompt, SAVE_DIR,return_image_data=True)[0]

##########################
# 2.3 Run the model
##########################

run(seed)