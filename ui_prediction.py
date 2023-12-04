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

import gradio as gr
from PIL import Image
import tempfile
import torch
import numpy as np

from zoedepth.utils.arg_utils import parse_unknown
import argparse
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config_user
import matplotlib
import cv2

from infer_user import regular_tile_param, random_tile_param
from zoedepth.models.base_models.midas import Resize
from torchvision.transforms import Compose
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from zoedepth.models.base_models.midas import Resize
from torchvision.transforms import Compose

import gradio as gr
import numpy as np
import trimesh
from zoedepth.utils.geometry import depth_to_points, create_triangles
from functools import partial
import tempfile

def depth_edges_mask(depth, occ_filter_thr):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    # mask = depth_grad > 0.05 # default in zoedepth
    mask = depth_grad > occ_filter_thr # preserve more edges (?)
    return mask

def load_state_dict(model, state_dict):
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
    return load_state_dict(model, ckpt)

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

config = get_config_user(args.model, **overwrite_kwargs)
config["pretrained_resource"] = ''
model = build_model(config)
model = load_ckpt(model, args.ckp_path)
model.eval()
model.cuda()

def colorize(value, cmap='magma_r', vmin=None, vmax=None):
    # normalize
    vmin = value.min() if vmin is None else vmin
    # vmax = value.max() if vmax is None else vmax
    vmax = np.percentile(value, 95) if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # ((1)xhxwx4)

    value = value[:, :, :3] # bgr -> rgb
    # rgb_value = value[..., ::-1]
    rgb_value = value

    return rgb_value

def predict_depth(model, image, mode, pn, reso, ps):

    pil_image = image
    image = transforms.ToTensor()(pil_image).unsqueeze(0).cuda()
    image_height, image_width = image.shape[-2], image.shape[-1]

    if reso != '':
        image_resolution = (int(reso.split('x')[0]), int(reso.split('x')[1]))
    else:
        image_resolution = (2160, 3840)
    image_hr = F.interpolate(image, image_resolution, mode='bicubic', align_corners=True)
    preprocess = Compose([Resize(512, 384, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")])
    image_lr = preprocess(image)

    if ps != '':
        patch_size = (int(ps.split('x')[0]), int(ps.split('x')[1]))
    else:
        patch_size = (int(image_resolution[0] // 4), int(image_resolution[1] // 4))

    avg_depth_map = regular_tile_param(
        model, 
        image_hr, 
        offset_x=0, 
        offset_y=0, 
        img_lr=image_lr,
        crop_size=patch_size, 
        img_resolution=image_resolution, 
        transform=preprocess,
        blr_mask=True)

    if mode== 'P16':
        pass
    elif mode== 'P49':
        regular_tile_param(
            model, 
            image_hr, 
            offset_x=patch_size[1]//2, 
            offset_y=0, 
            img_lr=image_lr, 
            iter_pred=avg_depth_map.average_map, 
            boundary=0, 
            update=True, 
            avg_depth_map=avg_depth_map, 
            crop_size=patch_size, 
            img_resolution=image_resolution, 
            transform=preprocess,
            blr_mask=True)
        regular_tile_param(
            model, 
            image_hr, 
            offset_x=0, 
            offset_y=patch_size[0]//2, 
            img_lr=image_lr, 
            iter_pred=avg_depth_map.average_map, 
            boundary=0, 
            update=True, 
            avg_depth_map=avg_depth_map, 
            crop_size=patch_size, 
            img_resolution=image_resolution, 
            transform=preprocess,
            blr_mask=True)
        regular_tile_param(
            model, 
            image_hr, 
            offset_x=patch_size[1]//2, 
            offset_y=patch_size[0]//2, 
            img_lr=image_lr, 
            iter_pred=avg_depth_map.average_map, 
            boundary=0, 
            update=True, 
            avg_depth_map=avg_depth_map, 
            crop_size=patch_size, 
            img_resolution=image_resolution, 
            transform=preprocess,
            blr_mask=True)
    elif mode == 'R':
        regular_tile_param(
            model, 
            image_hr, 
            offset_x=patch_size[1]//2, 
            offset_y=0, 
            img_lr=image_lr, 
            iter_pred=avg_depth_map.average_map, 
            boundary=0, 
            update=True, 
            avg_depth_map=avg_depth_map, 
            crop_size=patch_size, 
            img_resolution=image_resolution, 
            transform=preprocess,
            blr_mask=True)
        regular_tile_param(
            model, 
            image_hr, 
            offset_x=0, 
            offset_y=patch_size[0]//2, 
            img_lr=image_lr, 
            iter_pred=avg_depth_map.average_map, 
            boundary=0, 
            update=True, 
            avg_depth_map=avg_depth_map, 
            crop_size=patch_size, 
            img_resolution=image_resolution, 
            transform=preprocess,
            blr_mask=True)
        regular_tile_param(
            model, 
            image_hr, 
            offset_x=patch_size[1]//2, 
            offset_y=patch_size[0]//2, 
            img_lr=image_lr, 
            iter_pred=avg_depth_map.average_map, 
            boundary=0, 
            update=True, 
            avg_depth_map=avg_depth_map, 
            crop_size=patch_size, 
            img_resolution=image_resolution, 
            transform=preprocess,
            blr_mask=True)

        for i in range(int(pn)):
            random_tile_param(
                model, 
                image_hr, 
                img_lr=image_lr, 
                iter_pred=avg_depth_map.average_map, 
                boundary=0, 
                update=True, 
                avg_depth_map=avg_depth_map, 
                crop_size=patch_size, 
                img_resolution=image_resolution, 
                transform=preprocess,
                blr_mask=True)
    
    depth = avg_depth_map.average_map.detach().cpu()
    depth = F.interpolate(depth.unsqueeze(dim=0).unsqueeze(dim=0), (image_height, image_width), mode='bicubic', align_corners=True).squeeze().numpy()

    return depth

def create_demo(model):
    gr.Markdown("## Depth Prediction Demo")

    with gr.Accordion("Advanced options", open=False):
        mode = gr.Radio(["P49", "R"], label="Tiling mode", info="We recommand using P49 for fast evaluation and R with 1024 patches for best visualization results, respectively", elem_id='mode', value='R'),
        patch_number = gr.Slider(1, 1024, label="Please decide the number of random patches (Only useful in mode=R)", step=1, value=256)
        resolution = gr.Textbox(label="Proccessing resolution (Default 4K. Use 'x' to split height and width.)", elem_id='mode', value='2160x3840')
        patch_size = gr.Textbox(label="Patch size (Default 1/4 of image resolution. Use 'x' to split height and width.)", elem_id='mode', value='540x960')
    
    with gr.Row():
        input_image = gr.Image(label="Input Image", type='pil', elem_id='img-display-input')
        depth_image = gr.Image(label="Depth Map", elem_id='img-display-output')
    raw_file = gr.File(label="16-bit raw depth, multiplier:256")
    submit = gr.Button("Submit")

    def on_submit(image, mode, pn, reso, ps):
        depth = predict_depth(model, image, mode, pn, reso, ps)
        colored_depth = colorize(depth, cmap='gray_r')
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth = Image.fromarray((depth*256).astype('uint16'))
        raw_depth.save(tmp.name)
        return [colored_depth, tmp.name]
    
    submit.click(on_submit, inputs=[input_image, mode[0], patch_number, resolution, patch_size], outputs=[depth_image, raw_file])
    examples = gr.Examples(examples=["examples/example_1.jpeg", "examples/example_2.jpeg", "examples/example_3.jpeg"], inputs=[input_image])

def get_mesh(model, image, mode, pn, reso, ps, keep_edges, occ_filter_thr, fov):
    depth = predict_depth(model, image, mode, pn, reso, ps)

    image.thumbnail((1024,1024))  # limit the size of the input image
    depth = F.interpolate(torch.from_numpy(depth).unsqueeze(dim=0).unsqueeze(dim=0), (image.height, image.width), mode='bicubic', align_corners=True).squeeze().numpy()

    pts3d = depth_to_points(depth[None], fov=float(fov))
    pts3d = pts3d.reshape(-1, 3)

    # Create a trimesh mesh from the points
    # Each pixel is connected to its 4 neighbors
    # colors are the RGB values of the image

    verts = pts3d.reshape(-1, 3)
    image = np.array(image)
    if keep_edges:
        triangles = create_triangles(image.shape[0], image.shape[1])
    else:
        triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth, occ_filter_thr=float(occ_filter_thr)))
    colors = image.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

    # Save as glb
    glb_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    glb_path = glb_file.name
    mesh.export(glb_path)
    return glb_path

def create_demo_3d(model):

    gr.Markdown("### Image to 3D Mesh")
    gr.Markdown("Convert a single 2D image to a 3D mesh")

    with gr.Accordion("Advanced options", open=False):
        mode = gr.Radio(["P49", "R"], label="Tiling mode", info="We recommand using P49 for fast evaluation and R with 1024 patches for best visualization results, respectively", elem_id='mode', value='R'),
        patch_number = gr.Slider(1, 1024, label="Please decide the number of random patches (Only useful in mode=R)", step=1, value=256)
        resolution = gr.Textbox(label="Proccessing resolution (Default 4K. Use 'x' to split height and width)", value='2160x3840')
        patch_size = gr.Textbox(label="Patch size (Default 1/4 of image resolution. Use 'x' to split height and width)", value='540x960')

        checkbox = gr.Checkbox(label="Keep occlusion edges", value=False)
        # occ_filter_thr = gr.Textbox(label="Occlusion filter threshold", info="Larger value will reserve more edges (Only useful when NOT keeping occlusion edges)", value='0.5')
        # fov = gr.Textbox(label="FOV for inv-projection", value='55')

        occ_filter_thr = gr.Slider(0.01, 5, label="Occlusion edge filter threshold", info="Larger value will reserve more occlusion edges (Only useful when NOT keeping occlusion edges)", step=0.01, value=0.2)
        fov = gr.Slider(5, 180, label="FOV for inv-projection", step=1, value=55)


    with gr.Row():
        input_image = gr.Image(label="Input Image", type='pil')
        result = gr.Model3D(label="3d mesh reconstruction", clear_color=[1.0, 1.0, 1.0, 1.0])
    
    submit = gr.Button("Submit")
    submit.click(partial(get_mesh, model), inputs=[input_image, mode[0], patch_number, resolution, patch_size, checkbox, occ_filter_thr, fov], outputs=[result])
    examples = gr.Examples(examples=["examples/example_1.jpeg", "examples/example_4.jpeg", "examples/example_3.jpeg"], inputs=[input_image])

# controlnet

css = """
    #img-display-container {
        max-height: 50vh;
        }
    #img-display-input {
        max-height: 40vh;
        }
    #img-display-output {
        max-height: 40vh;
        }
"""

title = "# PatchFusion"
description = """Official demo for **PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation**.

PatchFusion is a deep learning model for high-resolution metric depth estimation from a single image.

Please refer to our [paper](https://arxiv.org/abs/2312.02284) or [github](https://github.com/zhyever/PatchFusion) for more details."""

with gr.Blocks(css=css) as demo:
    
    gr.Markdown(title)
    gr.Markdown(description)
    # create_demo(model)
    with gr.Tab("Depth Prediction"):
        create_demo(model)
    with gr.Tab("Image to 3D"):
        create_demo_3d(model)

    # gr.HTML('''<br><br><br><center>You can duplicate this Space to skip the queue:<a href="https://huggingface.co/spaces/shariqfarooq/ZoeDepth?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a><br>
    #     <p><img src="https://visitor-badge.glitch.me/badge?page_id=shariqfarooq.zoedepth_demo_hf" alt="visitors"></p></center>''')

if __name__ == '__main__':
    demo.queue().launch(share=True)