import torch
from torch.utils.data import Dataset
import cv2
import glob
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


class MSUDenseLeavesDataset(Dataset):

    def __init__(self, filepath, random_augmentation=False, augm_probability=0.2):
        self.filepath = filepath
        # each sample is made of image-labels-mask
        self.images = glob.glob(filepath + '*_img.png')
        self.labels = glob.glob(filepath + '*_label.png')
        self.masks = glob.glob(filepath + '*_mask.png')
        self.n_samples = len(self.images)

        self.augmentation = random_augmentation
        self.probability = augm_probability

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        # read image-labels-mask and return them
        image = cv2.imread(self.images[item])
        label = cv2.imread(self.labels[item])
        mask = cv2.imread(self.masks[item])

        # HxWxC-->CxHxW, bgr->rgb and tensor transformation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.transpose(2, 0, 1) / 255.)

        label = torch.from_numpy(label).long()
        mask = torch.from_numpy(mask).float()

        if self.augmentation:
            # augment data randomly
            if random.random() > self.probability:
                # rotate
                angle = random.randint(-45, 45)

                def rotate(img, angle):
                    img = TF.to_pil_image(img)
                    img = TF.rotate(img, angle)
                    # additionally flip image with some prob
                    flip = transforms.Compose([transforms.RandomVerticalFlip(0.2),
                                               transforms.RandomHorizontalFlip(0.4)])

                    return flip(img)

                image = rotate(image, angle)
                label = rotate(label, angle)
                mask = rotate(mask, angle)

        return image, label, mask

if __name__ == '__main__':
    dataset = MSUDenseLeavesDataset('', random_augmentation=False, augm_probability=1.0)
    for img, l, m in dataset:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.imshow('labels', l)
        cv2.waitKey(0)
        cv2.imshow('mask', m)
        cv2.waitKey(0)
        break