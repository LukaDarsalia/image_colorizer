from math import log10, sqrt
from sympy import O
import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np


def PSNR(original, colored):
    mse = torch.mean((original - colored) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(original, colored):
    return 0
    original = original.cpu().numpy()
    colored = colored.cpu().numpy()
    ssim_values = []
    for original_img, colored_img in zip(original, colored):
        ssim_values.append(ssim(original_img, colored_img, data_range=max(original_img.max(), colored_img.max()) - min(original_img.min(), colored_img.min()), multichannel=True, channel_axis=3))
    return np.mean(ssim_values)