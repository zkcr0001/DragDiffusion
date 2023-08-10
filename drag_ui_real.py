# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

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
from lora_utils import train_lora

def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

def mask_image(image, mask, color=[255,0,0], alpha=0.5):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    contours = cv2.findContours(np.uint8(deepcopy(mask)), cv2.RETR_TREE, 
                        cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return out

def train_lora_interface(original_image,
                         prompt,
                         model_path,
                         vae_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_rank,
                         progress=gr.Progress()):
    train_lora(
        original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_rank,
        progress)
    return "Training LoRA Done!"

def inference(source_image,
              image_with_clicks,
              mask,
              prompt,
              points,
              n_actual_inference_step,
              lam,
              n_pix_step,
              model_path,
              vae_path,
              lora_path,
              save_dir="./results"
    ):

    print(points)
    print(type(points))
    print(mask)
    print(mask.shape)
    print(type(source_image))
    print(source_image.shape)
    print(type(image_with_clicks))
    print(image_with_clicks.shape)

    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    model = DragPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    # initialize parameters
    seed = 42 # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = n_actual_inference_step
    args.guidance_scale = 1.0

    args.unet_feature_idx = [2]

    args.sup_res = 256

    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    args.lr = 0.01

    args.n_pix_step = n_pix_step
    print(args)
    full_h, full_w = source_image.shape[:2]

    source_image = preprocess_image(source_image, device)
    image_with_clicks = preprocess_image(image_with_clicks, device)

    # set lora
    if lora_path == "":
        print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(source_image,
                               prompt,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=n_actual_inference_step)

    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res, args.sup_res), mode="nearest")

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1] / full_h, point[0] / full_w]) * args.sup_res
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)

    init_code = invert_code
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    updated_init_code, updated_text_emb = drag_diffusion_update(model, init_code, t,
        handle_points, target_points, mask, args)

    # inference the synthesized image
    gen_image = model(prompt,
        prompt_embeds=updated_text_emb,
        latents=updated_init_code,
        guidance_scale=1.0,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=n_actual_inference_step
        )

    # save the original image, user editing instructions, synthesized image
    save_result = torch.cat([
        source_image * 0.5 + 0.5,
        torch.ones((1,3,512,25)).cuda(),
        image_with_clicks * 0.5 + 0.5,
        torch.ones((1,3,512,25)).cuda(),
        gen_image[0:1]
    ], dim=-1)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("""
        # Official Implementation of [DragDiffusion](https://arxiv.org/abs/2306.14435)
        """)

    with gr.Tab(label="Image"):
        with gr.Row():
            # input image
            original_image = gr.State(value=None) # store original image
            mask = gr.State(value=None) # store mask
            selected_points = gr.State([]) # store points
            length = 480
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Draw Mask</p>""")
                canvas = gr.Image(type="numpy", tool="sketch", label="Draw Mask", show_label=True, height=length, width=length) # for mask painting
                train_lora_button = gr.Button("Train LoRA")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Click Points</p>""")
                input_image = gr.Image(type="numpy", label="Click Points", show_label=True, height=length, width=length) # for points clicking
                undo_button = gr.Button("Undo point")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 25px">Editing Results</p>""")
                output_image = gr.Image(type="numpy", label="Editing Results", show_label=True, height=length, width=length)
                run_button = gr.Button("Run")

        # general parameters
        with gr.Row():
            prompt = gr.Textbox(label="Prompt")
            lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path")
            lora_status_bar = gr.Textbox(label="display LoRA training status")

        # algorithm specific parameters
        with gr.Accordion(label="Algorithm Parameters", open=False):
            with gr.Tab("Base Model Config"):
                with gr.Row():
                    local_models_dir = 'local_pretrained_models'
                    local_models_choice = \
                        [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
                    model_path = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                        label="Diffusion Model Path",
                        choices=["runwayml/stable-diffusion-v1-5"] + local_models_choice
                    )
                    vae_path = gr.Dropdown(value="default",
                        label="VAE choice",
                        choices=["default",
                        "stabilityai/sd-vae-ft-mse"] + local_models_choice
                    )

            with gr.Tab("Drag Parameters"):
                with gr.Row():
                    n_pix_step = gr.Number(value=40, label="n_pix_step", precision=0)
                    lam = gr.Number(value=0.1, label="lam")
                    n_actual_inference_step = gr.Number(value=40, label="n_actual_inference_step", precision=0)

            with gr.Tab("LoRA Parameters"):
                with gr.Row():
                    lora_step = gr.Number(value=200, label="LoRA training steps", precision=0)
                    lora_lr = gr.Number(value=0.0002, label="LoRA learning rate")
                    lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)

    # once user upload an image, the original image is stored in `original_image`
    # the same image is displayed in `input_image` for point clicking purpose
    def store_img(img):
        image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
        image = Image.fromarray(image)
        image = exif_transpose(image)
        # resize the input to 512x512
        # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        image = image.resize((512,512), PIL.Image.BILINEAR)
        image = np.array(image)
        mask  = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        if mask.sum() > 0:
            mask = np.uint8(mask > 0)
            masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
        else:
            masked_img = image.copy()
        # when new image is uploaded, `selected_points` should be empty
        return image, [], masked_img, mask
    canvas.edit(
        store_img,
        [canvas],
        [original_image, selected_points, input_image, mask]
    )

    # user click the image to get points, and show the points on the image
    def get_point(img, sel_pix, evt: gr.SelectData):
        # collect the selected point
        sel_pix.append(evt.index)
        # draw points
        points = []
        for idx, point in enumerate(sel_pix):
            if idx % 2 == 0:
                # draw a red circle at the handle point
                cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
            else:
                # draw a blue circle at the handle point
                cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
            points.append(tuple(point))
            # draw an arrow from handle point to target point
            if len(points) == 2:
                cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
                points = []
        return img if isinstance(img, np.ndarray) else np.array(img)
    input_image.select(
        get_point,
        [input_image, selected_points],
        [input_image],
    )

    # clear all handle/target points
    def undo_points(original_image, mask):
        if mask.sum() > 0:
            mask = np.uint8(mask > 0)
            masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
        else:
            masked_img = original_image.copy()
        return masked_img, []

    train_lora_button.click(
        train_lora_interface,
        [original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_rank],
        [lora_status_bar]
    )

    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, selected_points]
    )

    run_button.click(
        inference,
        [original_image,
        input_image,
        mask,
        prompt,
        selected_points,
        n_actual_inference_step,
        lam,
        n_pix_step,
        model_path,
        vae_path,
        lora_path,
        ],
        [output_image]
    )

demo.queue().launch(share=True, debug=True)
