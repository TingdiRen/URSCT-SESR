import torch
import cv2
from pytorch_msssim import ssim
from torchvision.transforms.functional import pad

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def torchSSIM(tar_img, prd_img):
    return ssim(tar_img, prd_img, data_range=1.0, size_average=True)

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def pad_img(img, padded_size):
    C, H, W = img.shape
    pad_size = [(padded_size[0]-H) // 2, (padded_size[1]-W) // 2]
    return pad(img, pad_size)