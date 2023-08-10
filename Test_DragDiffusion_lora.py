import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, AutoencoderKL
from drag_pipeline import DragPipeline

from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from drag_utils import drag_diffusion_update
from lora_utils import train_lora_api

image = np.array(Image.open("/home/ubuntu/frame_0000_square.png").resize((512,512)))
prompt = "a woman in CG style"
model_path = "runwayml/stable-diffusion-v1-5"
vae_path = "default"
save_lora_path = "./lora_tmp"
lora_step= 200
lora_lr = 0.0002
lora_rank = 16
train_lora_api(image, prompt, model_path, vae_path, save_lora_path, lora_step, lora_lr, lora_rank)

