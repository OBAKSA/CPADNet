import os
import shutil
import logging
import math
import numpy as np
import torch
import cv2
import glob


def refresh_folder(dir):
    """
    If directory does not exist, create.
    If directory exists, delete then create.
    """
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)


def save_checkpoint(path, denoiser, estimator, optimizer, iteration):
    torch.save(
        {
            'iteration': iteration,
            'denoiser_state_dict': denoiser.state_dict(),
            'estimator_state_dict': estimator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        path)


def save_checkpoint_single(path, denoiser, optimizer, scheduler, iteration):
    torch.save(
        {
            'iteration': iteration,
            'denoiser_state_dict': denoiser.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },
        path)


def get_psnr(img1, img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX = 1.0
    else:
        PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_psnr_uint16(img1, img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX = 1.0
    else:
        PIXEL_MAX = 65535.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d


def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out


def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()

    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def get_ssim(img1, img2, crop_border):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    ssims = []
    max_value = 1 if img1.max() <= 1 else 255
    with torch.no_grad():
        final_ssim = _ssim_3d(img1, img2, max_value)
        ssims.append(final_ssim)

    return np.array(ssims).mean()


class AverageMeter(object):
    """
    Keep track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def logger(log_adr, logger_name, mode='w'):
    """
    Logger
    """
    # create logger
    _logger = logging.getLogger(logger_name)
    # set level
    _logger.setLevel(logging.INFO)
    # set format
    formatter = logging.Formatter('%(message)s')
    # stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    # file
    file_handler = logging.FileHandler(log_adr, mode=mode)
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    return _logger


def date_time(secs):
    day = int(secs // (24 * 3600))
    secs = secs % (24 * 3600)
    hour = int(secs // 3600)
    secs %= 3600
    minutes = int(secs // 60)
    secs %= 60
    seconds = int(secs)
    return f'{day} d {hour} h {minutes} m {seconds} s'


##### camera parameter embedding #####
def embed_param(param: torch.Tensor, min_val, max_val) -> torch.Tensor:
    return torch.cat([
        (param / max_val),
        (param / max_val).sqrt(),
        (param / max_val) ** .25,
        (1. / (param / min_val)),
        (1. / (param / min_val)).sqrt(),
        (1. / (param / min_val)) ** .25,
        (param.log2() - math.log2(min_val)) / math.log2(max_val / min_val),
        (param.log2()).sin(),
        (param.log2()).cos()
    ], dim=1)
