from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import glob
import cv2
import rawpy
from skimage import io
from collections import defaultdict

test_meta_text_names_full_dir = './metadata/Sony_test_list.txt'


def build_dataset(directory):
    textfile = open(directory, 'r')
    lines = textfile.readlines()

    noisy_filename_list = []
    gt_filename_list = []

    exposure_time_list = []
    iso_list = []
    f_stop_list = []

    for line in lines:
        line_element = line.split(' ')
        noisy_filename = line_element[0].split('/')[-1]
        gt_filename = line_element[1].split('/')[-1]

        exposure_time = float(noisy_filename.split('_')[-1][:-5])
        iso = float(line_element[2][3:])
        f_stop = float(line_element[3].split('\n')[0][1:])

        noisy_filename_list.append(noisy_filename)
        gt_filename_list.append(gt_filename)
        exposure_time_list.append(exposure_time)
        iso_list.append(iso)
        f_stop_list.append(f_stop)

    return noisy_filename_list, gt_filename_list, exposure_time_list, iso_list, f_stop_list


#### SID ####
class ImageDataset_SID_Evaluation(Dataset):
    def __init__(self, noisy_dirs, gt_dirs):
        # image_dirs : has HQ/LQ directory
        # image_files : noisy image
        # targets : clean image

        self.noisy_dirs = noisy_dirs
        self.gt_dirs = gt_dirs

        noisy_files, gt_files, exposure_list, iso_list, fstop_list = build_dataset(test_meta_text_names_full_dir)

        self.noisy_files = noisy_files
        self.gt_files = gt_files
        self.exposure_list = exposure_list
        self.iso_list = iso_list
        self.fstop_list = fstop_list

        self.exposure_list = torch.as_tensor(self.exposure_list)
        self.iso_list = torch.as_tensor(self.iso_list)
        self.fstop_list = torch.as_tensor(self.fstop_list)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_name = self.noisy_files[idx]
        gt_name = self.gt_files[idx]
        exposure = self.exposure_list[idx]
        iso = self.iso_list[idx]
        fstop = self.fstop_list[idx]

        noisy = (cv2.cvtColor(cv2.imread(self.noisy_dirs + noisy_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)) / 255.
        gt = (cv2.cvtColor(cv2.imread(self.gt_dirs + gt_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)) / 255.

        if self.transform:
            noisy = self.transform(noisy)
            gt = self.transform(gt)

        sample = noisy, gt, exposure, iso, fstop, noisy_name

        return sample
