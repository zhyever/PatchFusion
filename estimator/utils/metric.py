import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import canny
import kornia

def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def shift_2d_replace(data, dx, dy, constant=False):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data

def soft_edge_error(pred, gt, radius=1):
    abs_diff=[]
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            abs_diff.append(np.abs(shift_2d_replace(gt, i, j, 0) - pred))
    return np.minimum.reduce(abs_diff)

def get_boundaries(disp, th=1., dilation=10):
    edges_y = np.logical_or(np.pad(np.abs(disp[1:, :] - disp[:-1, :]) > th, ((1, 0), (0, 0))),
                            np.pad(np.abs(disp[:-1, :] - disp[1:, :]) > th, ((0, 1), (0, 0))))
    edges_x = np.logical_or(np.pad(np.abs(disp[:, 1:] - disp[:, :-1]) > th, ((0, 0), (1, 0))),
                            np.pad(np.abs(disp[:, :-1] - disp[:,1:]) > th, ((0, 0), (0, 1))))
    edges = np.logical_or(edges_y,  edges_x).astype(np.float32)

    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges

def compute_metrics(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, disp_gt_edges=None, additional_mask=None):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            # pred, gt.shape[-2:], mode='bilinear', align_corners=True).squeeze()
            pred, gt.shape[-2:], mode='bilinear', align_corners=False).squeeze()

    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    eval_mask = np.ones(valid_mask.shape)
    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
            
    valid_mask = np.logical_and(valid_mask, eval_mask)
    
    # for prompt depth
    if additional_mask is not None:
        additional_mask = additional_mask.squeeze().detach().cpu().numpy()
        valid_mask = np.logical_and(valid_mask, additional_mask)
        
    metrics = compute_errors(gt_depth[valid_mask], pred[valid_mask])
        
    if disp_gt_edges is not None:
        
        edges = disp_gt_edges.squeeze().numpy()
        mask = valid_mask.squeeze() # squeeze
        mask = np.logical_and(mask, edges)

        see_depth = torch.tensor([0])
        if mask.sum() > 0:
            see_depth_map = soft_edge_error(pred, gt_depth)
            see_depth_map_valid = see_depth_map[mask]
            see_depth = see_depth_map_valid.mean()
        metrics['see'] = see_depth
    
    return metrics


def eps(x):
    """Return the `eps` value for the given `input` dtype. (default=float32 ~= 1.19e-7)"""
    dtype = torch.float32 if x is None else x.dtype
    return torch.finfo(dtype).eps

def to_log(depth):
    """Convert linear depth into log depth."""
    depth = torch.tensor(depth)
    depth = (depth > 0) * depth.clamp(min=eps(depth)).log()
    return depth

def to_inv(depth):
    """Convert linear depth into disparity."""
    depth = torch.tensor(depth)
    disp = (depth > 0) / depth.clamp(min=eps(depth))
    return disp

def extract_edges(depth,
                  preprocess=None,
                  sigma=1,
                  mask=None,
                  use_canny=True):
    """Detect edges in a dense LiDAR depth map.

    :param depth: (ndarray) (h, w, 1) Dense depth map to extract edges.
    :param preprocess: (str) Additional depth map post-processing. (log, inv, none)
    :param sigma: (int) Gaussian blurring sigma.
    :param mask: (Optional[ndarray]) Optional boolean mask of valid pixels to keep.
    :param use_canny: (bool) If `True`, use `Canny` edge detection, otherwise `Sobel`.
    :return: (ndarray) (h, w) Detected depth edges in the image.
    """
    if preprocess not in {'log', 'inv', 'none', None}:
        raise ValueError(f'Invalid depth preprocessing. ({preprocess})')

    depth = depth.squeeze()
    if preprocess == 'log':
        depth = to_log(depth)
    elif preprocess == 'inv':
        depth = to_inv(depth)
        depth -= depth.min()
        depth /= depth.max()
    else:
        depth = torch.tensor(depth)
        input_value = (depth > 0) * depth.clamp(min=eps(depth))
        # depth = torch.log(input_value) / torch.log(torch.tensor(1.9))
        # depth = torch.log(input_value) / torch.log(torch.tensor(1.9))
        depth = torch.log(input_value) / torch.log(torch.tensor(1.5))
        
    depth = depth.numpy()

    if use_canny:
        edges = canny(depth, sigma=sigma, mask=mask)
    else:
        raise NotImplementedError("Sobel edge detection is not implemented yet.")

    return edges
