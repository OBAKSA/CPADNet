from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import skimage.measure
import random
import os
import glob
import cv2
from skimage import io


def phone2num(val):
    if val == 'G4':
        val_num = 0
    elif val == 'GP':
        val_num = 1
    elif val == 'IP':
        val_num = 2
    elif val == 'N6':
        val_num = 3
    elif val == 'S6':
        val_num = 4
    else:
        print("!!!!!! ERROR !!!!!!")
    val_num = float(val_num)
    return val_num


#### SIDD ####
class ImageDataset_SIDD_Evaluation(Dataset):
    def __init__(self, image_dirs):
        # image_dirs : has HQ/LQ directory
        # image_files : noisy image
        # targets : clean image

        self.image_dirs = image_dirs

        self.noisy_files = glob.glob(self.image_dirs + '/*/*NOISY*.PNG')

        self.noisy_files.sort()

        self.phone_list = []
        self.iso_list = []
        self.shutter_list = []
        bv_iso = open('./metadata/sidd_valid_param.txt', 'r')
        while True:
            textline = bv_iso.readline()
            if textline == '':
                break
            else:
                phone_string = textline.split('\t')[0]
                iso_string = textline.split('\t')[1]
                shutter_string = textline.split('\t')[3]
                self.phone_list.append(int(phone2num(phone_string)))
                self.iso_list.append(float(iso_string))
                self.shutter_list.append(float(shutter_string))
        self.phone_list = torch.as_tensor(self.phone_list)
        self.iso_list = torch.as_tensor(self.iso_list)
        self.shutter_list = torch.as_tensor(self.shutter_list)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # self.transform = transform

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        img_name = self.noisy_files[idx]
        phone = self.phone_list[idx]
        iso = self.iso_list[idx]
        shutter = self.shutter_list[idx]

        image = (cv2.cvtColor(cv2.imread(img_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)) / 255

        height, width, _ = image.shape
        if height % 8 != 0:
            image = image[:height // 8 * 8, :, :]
        if width % 8 != 0:
            image = image[:, :width // 8 * 8, :]

        if self.transform:
            image = self.transform(image)

        sample = image, phone, iso, shutter, img_name

        return sample
