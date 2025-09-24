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
from archs.restormer.restormer_arch import Restormer_SIDD
from archs.restormer.restormer_arch_control import CPA_Restormer_SIDD
from model import BaseDenoiser_SIDD, CPADNet_SIDD
from data.data_SIDD import ImageDataset_SIDD_Evaluation

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
    parser.add_argument('--eval_path', type=str, default='./datasets/SIDD/SIDD_Benchmark_Data',
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
    test_dataset = ImageDataset_SIDD_Evaluation(image_dirs=opt.eval_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Model
    with torch.no_grad():
        if opt.exp_name == 'baseline':
            ckpt_name = 'SIDD_baseline.pth'
            denoiser = BaseDenoiser_SIDD(channels=32).cuda()
        elif opt.exp_name == 'cpadnet':
            ckpt_name = 'SIDD_cpadnet.pth'
            denoiser = CPADNet_SIDD(channels=32).cuda()
        elif opt.exp_name == 'restormer':
            ckpt_name = 'SIDD_restormer.pth'
            denoiser = Restormer_SIDD(dim=32).cuda()
        elif opt.exp_name == 'cpa_restormer':
            ckpt_name = 'SIDD_cpa_restormer.pth'
            denoiser = CPA_Restormer_SIDD(dim=32).cuda()

        # Load checkpoint
        ckpt = torch.load(f'./pretrained_models/{ckpt_name}', map_location=torch.device('cuda'))
        denoiser.load_state_dict(ckpt['denoiser_state_dict'])

        # Validate
        validate(opt, test_dataloader, denoiser, log)


def validate(opt, test_dataloader, denoiser, log):
    # Eval mode
    denoiser.eval()

    # Validate
    num_data = 0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            noisy = data[0].cuda()

            phone = data[1].cuda()
            iso = data[2].cuda()
            shutter = data[3].cuda()
            phone = phone.unsqueeze(0)
            iso = iso.unsqueeze(0)
            shutter = shutter.unsqueeze(0)
            image_name = data[4][0]
            image_name = image_name.split('/')[-1][:-4]
            print(image_name)
            print(phone)
            print(iso)
            print(shutter)
            print()

            # Denoiser output
            iso = embed_param(iso, 50, 10000)
            shutter = embed_param(shutter, 20, 8460)

            param = torch.cat([iso, shutter], dim=1)
            phone = phone.int()

            denoised = denoiser(noisy, param, phone)
            denoised = torch.clamp(denoised, 0., 1.)

            # Get PSNR and save results
            denoised = denoised.cpu().detach()
            denoised = (np.transpose(np.array(denoised)[0], (1, 2, 0)) * 65535).astype(np.uint16)
            noisy = noisy.cpu().detach()
            noisy = (np.transpose(np.array(noisy)[0], (1, 2, 0)) * 65535).astype(np.uint16)

            cv2.imwrite(f'./result/{opt.exp_name}/result_imgs/{image_name}.png',
                        cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))

            # Save GT and noisy
            if opt.save_gt_noisy:
                cv2.imwrite(f'./result/{opt.exp_name}/result_imgs/{str(idx).zfill(4)}_gt.png',
                            cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'./result/{opt.exp_name}/result_imgs/{str(idx).zfill(4)}_noisy.png',
                            cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))

            num_data += 1


if __name__ == '__main__':
    main()
