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
print(depth_prediction.shape)