import random
import numpy as np
import copy
from PIL import Image

def aug_flip(image, depth_gt):
    do_flip = random.random()
    if do_flip > 0.5:
        image = copy.deepcopy(image[:, ::-1, :])
        if isinstance(depth_gt, list):
            depth_gt_flipped = []
            for depth in depth_gt:
                depth_gt_flipped.append(copy.deepcopy(depth[:, ::-1]))
            depth_gt = depth_gt_flipped
        else: 
            depth_gt = copy.deepcopy(depth_gt[:, ::-1])
    return image, depth_gt

def aug_color(image, brightness_range=(0.9, 1.1)):
    do_augment = random.random()
    if do_augment > 0.5:
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        image = copy.deepcopy(image_aug)
    return image

def aug_rotate(image, depth_gt, degree, input_format='numpy'):
    random_angle = (random.random() - 0.5) * 2 * degree
    
    if input_format != 'PIL':
        image = Image.fromarray(image)
    image = image.rotate(random_angle, resample=Image.BILINEAR)
    if input_format != 'PIL':
        image = copy.deepcopy(np.asarray(image))
    else:
        image = copy.deepcopy(image)
    
    if isinstance(depth_gt, list):
        depth_gt_rot = []
        for depth in depth_gt:
            if input_format != 'PIL':
                depth = Image.fromarray(depth)
            depth = depth.rotate(random_angle, resample=Image.NEAREST)
            if input_format != 'PIL':
                depth_gt_rot.append(copy.deepcopy(np.asarray(depth)))
            else:
                depth_gt_rot.append(copy.deepcopy(depth))
        depth_gt = depth_gt_rot
    else:
        if input_format != 'PIL':
            depth_gt = Image.fromarray(depth_gt)
        depth_gt.rotate(random_angle, resample=Image.NEAREST)
        if input_format != 'PIL':
            depth_gt = np.asarray(depth_gt)
    
    return image, depth_gt

def random_crop(image, depth_gt, crop_size):
    c, h, w = image.shape
    h_start = random.randint(0, h - crop_size[0])
    w_start = random.randint(0, w - crop_size[1])
    image = copy.deepcopy(image[:, h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]])
    
    if isinstance(depth_gt, list):
        depth_crop = []
        for depth in depth_gt:
            depth_crop.append(copy.deepcopy(depth[:, h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]))
        depth_gt = depth_crop
    else: 
        depth_gt = copy.deepcopy(depth_gt[:, h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]])

    return image, depth_gt, [h_start, w_start]
