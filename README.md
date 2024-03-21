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

- TODO: We're refining the inference and training docs.
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

- Manually download the checkpoint from [here](https://huggingface.co/zhyever/PatchFusion/tree/main). For example, if we want to use depth-anything vitl, we need to download three checkpoints: [coarse_pretrain.pth](https://huggingface.co/zhyever/PatchFusion/blob/main/depthanything_vitl_u4k/coarse_pretrain/checkpoint_24.pth), [fine_pretrain.pth](https://huggingface.co/zhyever/PatchFusion/blob/main/depthanything_vitl_u4k/fine_pretrain/checkpoint_24.pth), and [patchfusion.pth](https://huggingface.co/zhyever/PatchFusion/blob/main/depthanything_vitl_u4k/patchfusion/checkpoint_16.pth). All training logs are provided there.

- Save them to the local folder. For example: we put them at: `./work_dir/depth-anything/ckps`

- Then, set the checkpoint path in the corresponding config files (e.g. `./configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py` in this case)

``` yaml
model.config.pretrain_model=['./work_dir/depth-anything/ckps/coarse_pretrain.pth', './work_dir/depth-anything/ckps/fine_pretrain.pth']

# Note the default path would be: './work_dir/depthanything_vitl_u4k/coarse_pretrain/checkpoint_24.pth', './work_dir/depthanything_vitl_u4k/fine_pretrain/checkpoint_24.pth'. Just look for them and replace them correspondingly.
```
- Lastly, load the model locally:
```python
from mmengine.config import Config
cfg_path = './configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py'
cfg = Config.fromfile(cfg_path) # load corresponding config for depth-anything vitl.
model = build_model(cfg.model) # build the model 
print(model.load_dict(torch.load(cfg.ckp_path)['model_state_dict']), logger='current') # load checkpoint
```
When building the PatchFusion model, it will load the coarse and fine checkpoints in the `init` function. Because the `patchfusion.pth` only contains the parameters of the fusion network, there will be some warnings here.

</details>

## **User Inference**

### Running:
```bash
python run.py <config> --ckp-path <checkpoints> --cai-mode <m1 | m2 | rn> --cfg-option general_dataloader.dataset.rgb_image_dir='<img-directory>' --save --work-dir <output-path> --test-type general [--gray-scale]
```
Arguments:
- `config`: `configs/patchfusion_depthanything/depthanything_general.py` and `./configs/patchfusion_zoedepth/zoedepth_general.py` are for Depth-Anything and ZoeDepth inference, respectively.
- `--ckp-path`: we can choose from `Zhyever/patchfusion_depth_anything_vits14`, `Zhyever/patchfusion_depth_anything_vitb14`, `/patchfusion_depth_anything_vitl14`, and `Zhyever/patchfusion_zoedepth`.
- `--cai-mode`: indicates the specific PatchFusion mode. `rn` means `n` patches in mode `r`.
- `--cfg-option`: specifies the input image directory. The prefix is indexing the config and just keep it there.
- `--work-dir`: saves the output files, including one colored depth map and one 16bit-png file (multiplier=256).
- `--gray-scale`: is set to save the grayscale depth map. Without it, by default, we apply a color palette to the depth map.

### Example:
```bash
python ./tools/test.py configs/patchfusion_depthanything/depthanything_general.py --ckp-path Zhyever/patchfusion_depth_anything_vitl14 --cai-mode r128 --cfg-option general_dataloader.dataset.rgb_image_dir='./examples/' --save --work-dir ./work_dir/predictions --test-type general
```

### Easy Way to Import PatchFusion:
<details>
<summary>Code snippet</summary>

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
default_resolution = model.tile_cfg['image_raw_shape']
image_resizer = model.resizer

image = cv2.imread('./examples/example_1.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
image = transforms.ToTensor()(np.asarray(image)) # raw image

image_lr = image_resizer(image.unsqueeze(dim=0)).float().to(DEVICE)
image_hr = F.interpolate(image.unsqueeze(dim=0), default_resolution, mode='bicubic', align_corners=True).float().to(DEVICE)

mode = 'r128'
process_num = 4 # batch process size. It could be larger if the GPU memory is larger
depth_prediction, _ = model(mode='infer', cai_mode=mode, process_num=process_num, image_lr=image_lr, image_hr=image_hr)
depth_prediction = F.interpolate(depth_prediction, image.shape[-2:])[0, 0].detach().cpu().numpy() # depth shape would be (h, w), similar to the input image.
```
</details>

### We provide more introductions about inference [here](./docs/user_infer.md). (TBD)

## **User Training**

### We provide more introductions about training [here](./docs/user_training.md). (TBD)

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