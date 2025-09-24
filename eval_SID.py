import warnings

warnings.filterwarnings('ignore')

from time import time
import numpy as np
import imageio
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from torch.utils.data import DataLoader

from utils import *
from archs.restormer.restormer_arch import Restormer_SID
from archs.restormer.restormer_arch_control import CPA_Restormer_SID
from model import BaseDenoiser_SID, CPADNet_SID
from data.data_SID import ImageDataset_SID_Evaluation

# Fix random seed for reproducibility
import random

random.seed(1994)
np.random.seed(1994)
torch.manual_seed(1994)
torch.cuda.manual_seed_all(1994)


def str2bool(s):
    return True if s.lower() == 'true' else False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default='1', help='evaluation batch size')
    parser.add_argument('--exp_name', type=str, default='cpadnet',
                        help='the name of experiment')
    parser.add_argument('--eval_path', type=str, default='./datasets/SID/Sony/Sony/',
                        help='evaluation file path')
    parser.add_argument('--save_gt_noisy', type=str2bool, default='False',
                        help='True for saving gt and noisy, else False')
    opt = parser.parse_args()

    # Make path to save results
    refresh_folder(f'./result/{opt.exp_name}/result_imgs')

    # Logger
    log = logger(f'./result/{opt.exp_name}/eval_log.txt', 'eval', 'w')
    opt_log = '-' * 15 + ' Options ' + '-' * 15 + '\n'
    for k, v in vars(opt).items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '-' * 39 + '\n'
    log.info(opt_log)

    # Dataset & Dataloader
    test_dataset = ImageDataset_SID_Test(image_dirs=opt.eval_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Model
    with torch.no_grad():
        if opt.exp_name == 'baseline':
            ckpt_name = 'SID_baseline.pth'
            denoiser = BaseDenoiser_SID(channels=32).cuda()
        elif opt.exp_name == 'cpadnet':
            ckpt_name = 'SID_cpadnet.pth'
            denoiser = CPADNet_SID(channels=32).cuda()
        elif opt.exp_name == 'restormer':
            ckpt_name = 'SID_restormer.pth'
            denoiser = Restormer_SID(dim=32).cuda()
        elif opt.exp_name == 'cpa_restormer':
            ckpt_name = 'SID_cpa_restormer.pth'
            denoiser = CPA_Restormer_SID(dim=32).cuda()

        # Load checkpoint
        ckpt = torch.load(f'./pretrained_models/{ckpt_name}', map_location=torch.device('cuda'))
        denoiser.load_state_dict(ckpt['denoiser_state_dict'])

        # Validate
        validate(opt, test_dataloader, denoiser, log)


def validate(opt, test_dataloader, denoiser, log):
    # Eval mode
    denoiser.eval()

    # Validate
    psnr_avg = 0.
    ssim_avg = 0.
    num_data = 0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            noisy = data[0].cuda()
            gt = data[1].cuda()

            exposure = data[2].cuda()
            iso = data[3].cuda()
            fstop = data[4].cuda()
            exposure = exposure.unsqueeze(0)
            iso = iso.unsqueeze(0)
            fstop = fstop.unsqueeze(0)
            image_name = data[5][0]
            print(image_name)
            print(exposure)
            print(iso)
            print(fstop)
            print()

            # Denoiser output
            exposure = embed_param(exposure, 0.033, 0.1)
            iso = embed_param(iso, 50, 25600)
            fstop = embed_param(fstop, 3.2, 22)

            param = torch.cat([exposure, iso, fstop], dim=1)

            denoised = denoiser(noisy, param)
            denoised = torch.clamp(denoised, 0., 1.)

            # Get PSNR and save results
            denoised = denoised.cpu().detach()
            denoised = (np.transpose(np.array(denoised)[0], (1, 2, 0)) * 255).astype(np.uint8)
            gt = gt.cpu().detach()
            gt = (np.transpose(np.array(gt)[0], (1, 2, 0)) * 255).astype(np.uint8)
            noisy = noisy.cpu().detach()
            noisy = (np.transpose(np.array(noisy)[0], (1, 2, 0)) * 255).astype(np.uint8)

            cv2.imwrite(f'./result/{opt.exp_name}/result_imgs/{str(idx).zfill(4)}_result.png',
                        cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))

            # Save GT and noisy
            if opt.save_gt_noisy:
                cv2.imwrite(f'./result/{opt.exp_name}/result_imgs/{str(idx).zfill(4)}_gt.png',
                            cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'./result/{opt.exp_name}/result_imgs/{str(idx).zfill(4)}_noisy.png',
                            cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))

            psnr = get_psnr(denoised, gt)
            log.info(f'{str(idx).zfill(4)}.png: {psnr:.4f}')
            psnr_avg += psnr

            ssim = get_ssim(denoised, gt, crop_border=0)
            # log.info(f'{str(idx * opt.batch_size + n).zfill(4)}.png: {ssim:.4f}')
            ssim_avg += ssim

            num_data += 1

        psnr_avg /= num_data
        log.info('-' * 40)
        log.info(f'Average PSNR: {psnr_avg:.4f}')

        ssim_avg /= num_data
        log.info('-' * 40)
        log.info(f'Average SSIM: {ssim_avg:.4f}')


if __name__ == '__main__':
    main()
