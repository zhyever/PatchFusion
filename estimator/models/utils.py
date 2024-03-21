import torch
import cv2
import numpy as np
import torch.nn.functional as F

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class HookTool: 
    def __init__(self):
        self.feat = None 

    def hook_in_fun(self, module, fea_in, fea_out):
        self.feat = fea_in
        
    def hook_out_fun(self, module, fea_in, fea_out):
        self.feat = fea_out

class RunningAverageMap:
    """ Saving avg depth estimation results."""
    def __init__(self, average_map, count_map):
        self.average_map = average_map
        self.count_map = count_map
        self.average_map = self.average_map / self.count_map

    def update(self, pred_map, ct_map):
        self.average_map = (pred_map + self.count_map * self.average_map) / (self.count_map + ct_map)
        self.count_map = self.count_map + ct_map
        
    def resize(self, resolution):
        temp_avg_map = self.average_map.unsqueeze(0).unsqueeze(0)
        temp_count_map = self.count_map.unsqueeze(0).unsqueeze(0)
        self.average_map = F.interpolate(temp_avg_map, size=resolution).squeeze()
        self.count_map = F.interpolate(temp_count_map, size=resolution, mode='bilinear', align_corners=True).squeeze()

def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.1*size[0]):size[0] - int(0.1*size[0]), int(0.1*size[1]): size[1] - int(0.1*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask