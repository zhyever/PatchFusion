<div align="center">
<h1>PatchFusion </h1>
<h3>An End-to-End Tile-Based Framework <br> for High-Resolution Monocular Metric Depth Estimation</h3>

[![Website](assets/badge-website.svg)](https://zhyever.github.io/patchfusion/) [![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2312.02284) [![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/zhyever/PatchFusion) [![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/zhyever/PatchFusion) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<a href="https://zhyever.github.io/">Zhenyu Li</a>, <a href="https://shariqfarooq123.github.io/">Shariq Farooq Bhat</a>, <a href="https://peterwonka.net/">Peter Wonka</a>. 
<br>KAUST

<center>
<img src='assets/showcase_3.gif'>
</center>

</div>

## ✨ **NEWS**

- 2024-03-25: Release [inference introduction](./docs/user_infer.md) and [training introduction](./docs/user_training.md).
- 2024-03-21: Release refactored codes (main branch in this repo).
- 2024-03-16: Release updated [huggingface demo](https://huggingface.co/spaces/zhyever/PatchFusion), which supports [Depth-Anything](https://github.com/LiheYoung/Depth-Anything).
- 2024-03-04: Accepted to CVPR 2024.
- 2023-12-12: Initially release [project page](https://zhyever.github.io/patchfusion/), [paper](https://arxiv.org/abs/2312.02284), [codes](certain_branch), and [demo](https://huggingface.co/spaces/zhyever/PatchFusion). Checkout `2d87adc9`.


## **Repo Features**

- 2024-03-21: Add support for [Depth-Anything](https://github.com/LiheYoung/Depth-Anything).
- 2024-03-21: Add support for customized tiling configurations.
- 2024-03-21: Add support for processing patches in a batch manner.
- 2024-03-21: Simplify training and inference pipeline.
- 2024-03-21: Bump Pytorch versions to 2.1.2.
- 2023-12-12: Release basic support for [ZoeDepth](https://github.com/isl-org/ZoeDepth).

 
## **Environment setup**

Install environment using `environment.yml` : 

Using [mamba](https://github.com/mamba-org/mamba) (fastest):
```bash
mamba env create -n patchfusion --file environment.yml
mamba activate patchfusion
```
Using conda : 

```bash
conda env create -n patchfusion --file environment.yml
conda activate patchfusion
```

### NOTE:
Before running the code, please first run:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/the/folder/PatchFusion"
export PYTHONPATH="${PYTHONPATH}:/path/to/the/folder/PatchFusion/external"
```
**Make sure that you have exported the `external` folder which stores codes from other repos (ZoeDepth, Depth-Anything, etc.)**

## **Pre-Train Model**

We provide PatchFusion with various base depth models: ZoeDepth-N, Depth-Anything-vits, Depth-Anything-vitb, and Depth-Anything-vitl. The inference time of PatchFusion is linearly related to the base model's inference time.

``` python
from estimator.models.patchfusion import PatchFusion
model_name = 'Zhyever/patchfusion_depth_anything_vitl14'

# valid model name:
# 'Zhyever/patchfusion_depth_anything_vits14', 
# 'Zhyever/patchfusion_depth_anything_vitb14', 
# 'Zhyever/patchfusion_depth_anything_vitl14', 
# 'Zhyever/patchfusion_zoedepth'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PatchFusion.from_pretrained(model_name).to(DEVICE).eval()
```

### *Without Network Connection Solution*

<details>
<summary>Click here for solutions</summary>

- Manually download the checkpoint from [here](https://huggingface.co/zhyever/PatchFusion/tree/main). For example, if you want to use depth-anything vitl, you need to download three checkpoints: [coarse_pretrain.pth](https://huggingface.co/zhyever/PatchFusion/blob/main/depthanything_vitl_u4k/coarse_pretrain/checkpoint_24.pth), [fine_pretrain.pth](https://huggingface.co/zhyever/PatchFusion/blob/main/depthanything_vitl_u4k/fine_pretrain/checkpoint_24.pth), and [patchfusion.pth](https://huggingface.co/zhyever/PatchFusion/blob/main/depthanything_vitl_u4k/patchfusion/checkpoint_16.pth).

- Save them to the local folder. For example: put them at: `./work_dir/depth-anything/ckps`

- Then, set the checkpoint path in the corresponding config files (e.g. `./configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py` in this case): 

``` yaml
model.config.pretrain_model=['./work_dir/depth-anything/ckps/coarse_pretrain.pth', './work_dir/depth-anything/ckps/fine_pretrain.pth']

# Note the default path would be: './work_dir/depthanything_vitl_u4k/coarse_pretrain/checkpoint_24.pth', './work_dir/depthanything_vitl_u4k/fine_pretrain/checkpoint_24.pth'. Just look for this item replace it correspondingly.
```

- Lastly, load the model locally:
```python
from mmengine.config import Config
cfg_path = './configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py'
cfg = Config.fromfile(cfg_path) # load corresponding config for depth-anything vitl.
model = build_model(cfg.model) # build the model 
print(model.load_dict(torch.load(cfg.ckp_path)['model_state_dict']), logger='current') # load checkpoint
```
When building the PatchFusion model, it will load the coarse and fine checkpoints in the `init` function. Because the `patchfusion.pth` only contains the parameters of the fusion network, there will be some warnings here. But it's totally fine. The idea is to save coarse model, fine model, and fusion model separately.

- We list the corresponding config path as below. Please make sure looking for the correct one before starting to modify.

| Model Name  | Config Path  | 
|---|---|
| Depth-Anything-vitl  |  `./configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py` |
| Depth-Anything-vitb  |  `./configs/patchfusion_depthanything/depthanything_vitb_patchfusion_u4k.py` |
| Depth-Anything-vits  |  `./configs/patchfusion_depthanything/depthanything_vits_patchfusion_u4k.py` |
| ZoeDepth-N  |  `./configs/patchfusion_zoedepth/zoedepth_patchfusion_u4k.py` |

</details>

## **User Inference**

### Running:
To execute user inference, use the following command:

```bash
python run.py ${CONFIG_FILE} --ckp-path <checkpoints> --cai-mode <m1 | m2 | rn> --cfg-option general_dataloader.dataset.rgb_image_dir='<img-directory>' [--save] --work-dir <output-path> --test-type general [--gray-scale] --image-raw-shape [h w] --patch-split-num [h, w]
```
Arguments Explanation (More details can be found [here](./docs/user_infer.md)):
- `${CONFIG_FILE}`: Select the configuration file from the following options based on the inference type you want to run:
    - `configs/patchfusion_depthanything/depthanything_general.py` for Depth-Anything
    - `configs/patchfusion_zoedepth/zoedepth_general.py` for ZoeDepth inference
- `--ckp-path`: Specify the checkpoint path. Select from the following options to load from Hugging Face (an active network connection is required). Local checkpoint files can also be used:
    - `Zhyever/patchfusion_depth_anything_vits14`
    - `Zhyever/patchfusion_depth_anything_vitb14`
    - `Zhyever/patchfusion_depth_anything_vitl14`
    - `Zhyever/patchfusion_zoedepth`
- `--cai-mode`: Define the specific PatchFusion mode to use. For example, rn indicates n patches in mode r.
- `--cfg-option`: Specify the input image directory. Maintain the prefix as it indexes the configuration. (To learn more about this, please refer to [MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html). Basically, we use MMEngine to organize the configurations of this repo).
- `--save`: Enable the saving of output files to the specified `--work-dir` directory (Make sure using it, otherwise there will be nothing saved).
- `--work-dir`: Directory where the output files will be stored, including a colored depth map and a 16-bit PNG file (multiplier=256).
- `--gray-scale`: If set, the output will be a grayscale depth map. If omitted, a color palette is applied to the depth map by default.
- `--image-raw-shape`: Specify the original dimensions of the input image. Input images will be resized to this resolution before being processed by PatchFusion. Default: `2160 3840`.
- `--patch-split-num`: Define how the input image is divided into smaller patches for processing. Default: `4 4`. ([Check more introductions](./docs/user_infer.md))
- `--process-num`: Batch size when processing the inference patchs. It will speed up the inference with the cost of gpu memory when increasing the number. Default: `2`.

### Example Usage:
Below is an example command that demonstrates how to run the inference process:
```bash
python ./tools/test.py configs/patchfusion_depthanything/depthanything_general.py --ckp-path Zhyever/patchfusion_depth_anything_vitl14 --cai-mode r32 --cfg-option general_dataloader.dataset.rgb_image_dir='./examples/' --save --work-dir ./work_dir/predictions --test-type general --image-raw-shape 1080 1920 --patch-split-num 2 2
```
This example performs inference using the `depthanything_general.py` configuration for Depth-Anything, loads the specified checkpoint `patchfusion_depth_anything_vitl14`, sets the PatchFusion mode to `r32`, specifies the input image directory `./examples/`, and saves the output to ./work_dir/predictions `./work_dir/predictions`. The original dimensions of the input image is `1080x1920` and the input image is divided into `2x2` patches.

### Easy Way to Import PatchFusion:
<details>
<summary>Code snippet</summary>

You can find this code snippet in `./tools/test_single_forward.py`.

```python
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from estimator.models.patchfusion import PatchFusion

model_name = 'Zhyever/patchfusion_depth_anything_vitl14'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PatchFusion.from_pretrained(model_name).to(DEVICE).eval()
image_raw_shape = model.tile_cfg['image_raw_shape']
image_resizer = model.resizer

image = cv2.imread('./examples/example_1.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
image = transforms.ToTensor()(np.asarray(image)) # raw image

image_lr = image_resizer(image.unsqueeze(dim=0)).float().to(DEVICE)
image_hr = F.interpolate(image.unsqueeze(dim=0), image_raw_shape, mode='bicubic', align_corners=True).float().to(DEVICE)

mode = 'r128' # inference mode
process_num = 4 # batch process size
depth_prediction, _ = model(mode='infer', cai_mode=mode, process_num=process_num, image_lr=image_lr, image_hr=image_hr)
depth_prediction = F.interpolate(depth_prediction, image.shape[-2:])[0, 0].detach().cpu().numpy() # depth shape would be (h, w), similar to the input image
```
</details>

### More introductions about inference are provided [here](./docs/user_infer.md).

## **User Training**

### Please refer to [user_training](./docs/user_training.md) for more details.

## **Acknowledgement**

We would like to thank [AK(@_akhaliq)](https://twitter.com/_akhaliq) and [@hysts](https://huggingface.co/hysts) from the HuggingFace team for the help.

## Citation
If you find our work useful for your research, please consider citing the paper
```
@article{li2023patchfusion,
    title={PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation}, 
    author={Zhenyu Li and Shariq Farooq Bhat and Peter Wonka},
    booktitle={CVPR},
    year={2024}
}
```