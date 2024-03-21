
1. Put your images in folder ``path/to/your/folder``

2. Run codes (with 4 gpus):
    ```bash
    bash ./tools/dist_test.sh configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py 4 --ckp-path Zhyever/patchfusion_depth_anything_vitl14 --test-type general --cai-mode m2 --cfg-option general_dataloader.dataset.rgb_image_dir='/ibex/ai/home/liz0l/codes/datasets/eth3d_test' --save --work-dir ./work_dir/show_images/eth3d_vitl
    ```

3. Check visualization results in ``path/to/show`` and depth results in ``path/to/save``, respectively.

**Args**
- `config`: Please use `./configs/patchfusion_depthanything/depthanything_general.py`, and `./configs/patchfusion_zoedepth/zoedepth_general.py` for Depth-Anything and ZoeDepth inference, respectively.
- `gpu_number`: We support multi-gpu inference. Here we use 4.
- `--ckp-path`: Valid model name are `Zhyever/patchfusion_depth_anything_vits14`, `Zhyever/patchfusion_depth_anything_vitb14`, `Zhyever/patchfusion_depth_anything_vitl14`, and `Zhyever/patchfusion_zoedepth`.