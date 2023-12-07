# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Zhenyu Li

from ControlNet.share import *
import einops
import torch
import random

import ControlNet.config as config
from pytorch_lightning import seed_everything
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler

import gradio as gr
import torch
import numpy as np
from zoedepth.utils.arg_utils import parse_unknown
import argparse
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config_user
import gradio as gr

from ui_prediction import predict_depth
import torch.nn.functional as F

model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./nfs/control_sd15_depth.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def depth_load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

    DataParallel prefixes state_dict keys with 'module.' when saving.
    If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
    If the model is a DataParallel model but the state_dict is not, then prefixes are added.
    """
    state_dict = state_dict.get('model', state_dict)
    # if model is a DataParallel model, then state_dict keys are prefixed with 'module.'

    do_prefix = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not do_prefix:
            k = k[7:]

        if not k.startswith('module.') and do_prefix:
            k = 'module.' + k

        state[k] = v

    model.load_state_dict(state, strict=True)
    print("Loaded successfully")
    return model

def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return depth_load_state_dict(model, ckpt)

def load_ckpt(model, checkpoint):
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--ckp_path", type=str, required=True)
parser.add_argument("-m", "--model", type=str, default="zoedepth")
parser.add_argument("--model_cfg_path", type=str, default="")
args, unknown_args = parser.parse_known_args()

overwrite_kwargs = parse_unknown(unknown_args)
overwrite_kwargs['model_cfg_path'] = args.model_cfg_path
overwrite_kwargs["model"] = args.model

config_depth = get_config_user(args.model, **overwrite_kwargs)
config_depth["pretrained_resource"] = ''
depth_model = build_model(config_depth)
depth_model = load_ckpt(depth_model, args.ckp_path)
depth_model.eval()
depth_model.cuda()

# controlnet
title = "# PatchFusion"
description = """Official demo for **PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation**.

PatchFusion is a deep learning model for high-resolution metric depth estimation from a single image.

Please refer to our [paper](???) or [github](???) for more details."""

def rescale(A, lbound=-1, ubound=1):
    """
    Rescale an array to [lbound, ubound].

    Parameters:
    - A: Input data as numpy array
    - lbound: Lower bound of the scale, default is 0.
    - ubound: Upper bound of the scale, default is 1.

    Returns:
    - Rescaled array
    """
    A_min = np.min(A)
    A_max = np.max(A)
    return (ubound - lbound) * (A - A_min) / (A_max - A_min) + lbound

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mode, patch_number, resolution, patch_size):
    with torch.no_grad():
        w, h = input_image.size
        detected_map = predict_depth(depth_model, input_image, mode, patch_number, resolution, patch_size)
        detected_map = F.interpolate(torch.from_numpy(detected_map).unsqueeze(dim=0).unsqueeze(dim=0), (image_resolution, image_resolution), mode='bicubic', align_corners=True).squeeze().numpy()

        H, W = detected_map.shape
        detected_map_temp = ((1 - detected_map / np.max(detected_map)) * 255)
        detected_map = detected_map_temp.astype("uint8")

        detected_map_temp = detected_map_temp[:, :, None]
        detected_map_temp = np.concatenate([detected_map_temp, detected_map_temp, detected_map_temp], axis=2)
        detected_map = detected_map[:, :, None]
        detected_map = np.concatenate([detected_map, detected_map, detected_map], axis=2)
        
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255)

        results = [x_samples[i] for i in range(num_samples)]

        return_list = [detected_map_temp] + results
        update_return_list = []
        for r in return_list:
            t_r = torch.from_numpy(r).unsqueeze(dim=0).permute(0, 3, 1, 2)
            t_r = F.interpolate(t_r, (h, w), mode='bicubic', align_corners=True).squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
            update_return_list.append(t_r)

    return update_return_list

title = "# PatchFusion"
description = """Official demo for **PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation**.

PatchFusion is a deep learning model for high-resolution metric depth estimation from a single image.

Please refer to our [paper](https://arxiv.org/abs/2312.02284) or [github](https://github.com/zhyever/PatchFusion) for more details."""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Depth Maps")
    with gr.Row():
        with gr.Column():
            # input_image = gr.Image(source='upload', type="pil")
            input_image = gr.Image(label="Input Image", type='pil')
            prompt = gr.Textbox(label="Prompt (input your description)", value='An evening scene with the Eiffel Tower, the bridge under the glow of street lamps and a twilight sky')
            run_button = gr.Button("Run")
            with gr.Accordion("Advanced options", open=False):
                # mode = gr.Radio(["P49", "R"], label="Tiling mode", info="We recommand using P49 for fast evaluation and R with 1024 patches for best visualization results, respectively", elem_id='mode', value='R'),
                mode = gr.Radio(["P49", "R"], label="Tiling mode", info="We recommand using P49 for fast evaluation and R with 1024 patches for best visualization results, respectively", elem_id='mode', value='R'),
                patch_number = gr.Slider(1, 1024, label="Please decide the number of random patches (Only useful in mode=R)", step=1, value=256)
                resolution = gr.Textbox(label="PatchFusion proccessing resolution (Default 4K. Use 'x' to split height and width.)", elem_id='mode', value='2160x3840')
                patch_size = gr.Textbox(label="Patch size (Default 1/4 of image resolution. Use 'x' to split height and width.)", elem_id='mode', value='540x960')

                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="ControlNet image resolution", minimum=256, maximum=2048, value=1024, step=64)
                strength = gr.Slider(label="Control strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                # detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
                ddim_steps = gr.Slider(label="steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="guidance scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative prompt", value='worst quality, low quality, lose details')
        with gr.Column():
            # result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mode[0], patch_number, resolution, patch_size]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    examples = gr.Examples(examples=["examples/example_2.jpeg", "examples/example_4.jpeg", "examples/example_5.jpeg"], inputs=[input_image])


if __name__ == '__main__':
    demo.queue().launch(share=True)