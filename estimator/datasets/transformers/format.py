
import torch
import numpy as np

def to_tensor(pic):
    if isinstance(pic, np.ndarray):
        if len(pic.shape) == 3:
            pic = torch.from_numpy(pic.transpose((2, 0, 1))) # img here
            # pic = torch.tensor(pic).permute(2, 0, 1)
        else:
            pic = torch.from_numpy(pic[np.newaxis, ...]) # depth map here
            # pic = torch.tensor(pic).unsqueeze(dim=0)
    
        return pic
    
    else:
        return pic