
# **User Training**

We provide training illustrations on Unreal4kStereo dataset in this document. Users can adopt to custome datasets based on this Unreal4kStereo version. We provide all configs for Depth-Anything and ZoeDepth training.


## Dataset Preparation:

Download the dataset from https://github.com/fabiotosi92/SMD-Nets.

Preprocess the dataset following the [instruction](https://github.com/fabiotosi92/SMD-Nets?tab=readme-ov-file#unrealstereo4k) (convert images to `raw` format).

Copy the split files in `./splits/u4k` and organize the folder structure as:

```none
monocular-depth-estimation-toolbox
├── estimator
├── docs
├── ...
├── data (it's included in `.gitignore`)
│   ├── u4k (recommand ln -s)
│   │   ├── 00000
│   │   │   ├── Disp0
│   │   │   │   ├── 00000.npy
│   │   │   │   ├── 00001.npy
│   │   │   │   ├── ...
│   │   │   ├── Extrinsics0
│   │   │   ├── Extrinsics1
│   │   │   ├── Image0
│   │   │   │   ├── 00000.raw (Note it's important to convert png to raw to speed up training)
│   │   │   │   ├── 00001.raw
│   │   │   │   ├── ...
│   │   ├── 00001
│   │   │   ├── Disp0
│   │   │   ├── Extrinsics0
│   │   │   ├── Extrinsics1
│   │   │   ├── Image0
|   |   ├── ...
|   |   ├── 00008
|   |   ├── splits
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   │   ├── test.txt
│   │   │   ├── test_out.txt
```

## Pre-trained Model Preparation:

Before trainig, please download pre-trained metric depth estimators from https://huggingface.co/zhyever/PatchFusion/tree/main. We provide pre-trained checkpoints for Depth-Anything and ZoeDepth.

| Model Name  | Config Path  | 
|---|---|
| Depth-Anything-vitl  |  `https://huggingface.co/zhyever/PatchFusion/blob/main/DepthAnything_vitl.pt` |
| Depth-Anything-vitb  |  `https://huggingface.co/zhyever/PatchFusion/blob/main/DepthAnything_vitb.pt` |
| Depth-Anything-vits  |  `https://huggingface.co/zhyever/PatchFusion/blob/main/DepthAnything_vits.pt` |
| ZoeDepth-N  |  `https://huggingface.co/zhyever/PatchFusion/blob/main/patchfusion_u4k.pt` |

Note that these checkpoints are pre-trained using the [offical implementation](https://github.com/isl-org/ZoeDepth), and as a result, their file names end with `.pt`. 

Put them to one specific folder, for example, `./work_dir/DepthAnything_vitl.pt`. (**Note:** `./work_dir` is included in `.gitignore`)

## Model Training:

This repo follows the design of [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox), but it's more flexible in training and inference. The overall training script follows this line of command:

``` bash
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Arguments Explanation:
- `${CONFIG_FILE}`: Select the configuration file for training
- `${GPU_NUM}`: Specify the number of GPU used for training (We use 4 as default)
- `[optional arguments]`: You can specify more arguments. We present some important arguments here
    - `--log-name`: experiment name shown in wandb website
    - `--work-dir`: `work-dir + log-name` indicates the path to save logs and checkpoints
    - `--tag`: tags shown in wandb website
    - `--debug`: if set, omit wandb log

You will see examples in the following sections.

Training PatchFusion includes three steps. Here, we use take DepthAnything vitl as an example. 

### Coarse Model Training

First, check the config file: `./configs/patchfusion_depthanything/depthanything_vitl_coarse_pretrain_u4k`. Modify the config item `zoe_depth_config.pretrained_resource` to the checkpoint path (the default path is `local::./work_dir/DepthAnything_vitl.pt`). The prefix `local::` is necessary because we based on the offical implementation.

Then, run:
``` bash
bash ./tools/dist_train.sh configs/patchfusion_depthanything/depthanything_vitl_coarse_pretrain_u4k.py 4 --work-dir ./work_dir/depthanything_vitl_u4k --log-name coarse_pretrain --tag coarse,da,vitl
```

As for this command, we will use the config `depthanything_vitl_coarse_pretrain_u4k.py`, 4 gpus to train the model, and save the logs and checkpoints to `./work_dir/depthanything_vitl_u4k/coarse_pretrain`. During training, you can check logs with experiment name `coarse_pretrain` on wandb.

### Fine Model Training

Again, check the config file: `./configs/patchfusion_depthanything/depthanything_vitl_fine_pretrain_u4k.py`. Modify the config item `zoe_depth_config.pretrained_resource` to the checkpoint path (the default path is `local::./work_dir/DepthAnything_vitl.pt`).

Then, run:
``` bash
bash ./tools/dist_train.sh configs/patchfusion_depthanything/depthanything_vitl_fine_pretrain_u4k.py 4 --work-dir ./work_dir/depthanything_vitl_u4k --log-name fine_pretrain --tag fine,da,vitl
```

### Fusion Model Training

Finally, we can train the fusion model. Check the config file: `./configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py`. Now, you need to modify the config item `model.config.pretrain_model` to the checkpoint paths of both the pre-trained coarse and fine models. (the default path is `['./work_dir/depthanything_vitl_u4k/coarse_pretrain/checkpoint_24.pth', './work_dir/depthanything_vitl_u4k/fine_pretrain/checkpoint_24.pth']`)

Finally, run:
``` bash
bash ./tools/dist_train.sh configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py 4 --work-dir ./work_dir/depthanything_vitl_u4k --log-name patchfusion --tag patchfusion,da,vitl
```

### Models & Configs

| Model Name  | Coarse Config  | Fine Config  | PatchFusion Config  | 
|---|---|---|---|
| Depth-Anything-vitl  | `./configs/patchfusion_depthanything/depthanything_vitl_coarse_pretrain_u4k.py` | `./configs/patchfusion_depthanything/depthanything_vitl_fine_pretrain_u4k.py` | `./configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py` |
| Depth-Anything-vitb  | `./configs/patchfusion_depthanything/depthanything_vitb_coarse_pretrain_u4k.py` | `./configs/patchfusion_depthanything/depthanything_vitb_fine_pretrain_u4k.py` | `./configs/patchfusion_depthanything/depthanything_vitb_patchfusion_u4k.py` |
| Depth-Anything-vits  | `./configs/patchfusion_depthanything/depthanything_vits_coarse_pretrain_u4k.py` | `./configs/patchfusion_depthanything/depthanything_vits_fine_pretrain_u4k.py` | `./configs/patchfusion_depthanything/depthanything_vits_patchfusion_u4k.py` |
| ZoeDepth-N  |  `./configs/patchfusion_zoedepth/zoedepth_coarse_pretrain_u4k.py` | `./configs/patchfusion_zoedepth/zoedepth_fine_pretrain_u4k.py` | `./configs/patchfusion_zoedepth/zoedepth_patchfusion_u4k.py` |

## Model Validation:

### Running Time Validation
During training, the validation is processed intermittently. In the config file, you can change the related settings. For example, 

``` bash
train_cfg=dict(max_epochs=16, val_interval=2, save_checkpoint_interval=16, log_interval=100, train_log_img_interval=500, val_log_img_interval=50, val_type='epoch_base', eval_start=0)
```

By changing `val_interval=4`, you can validate every 4 epochs. 

### Offline Validation
Run:

```bash
bash ./tools/dist_test.sh configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py 4 --ckp-path ./work_dir/depthanything_vitl_u4k/patchfusion/checkpoint_16.pth --cai-mode m1
```

Check more details of arguments at [Inference with Multiple GPUs](https://github.com/zhyever/PatchFusion/blob/main/docs/user_infer.md#inference-with-multiple-gpus) and [Running](https://github.com/zhyever/PatchFusion?tab=readme-ov-file#running).




