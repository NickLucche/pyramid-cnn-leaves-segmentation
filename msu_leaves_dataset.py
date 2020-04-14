import torch
from torch.utils.data import Dataset
import cv2
import glob
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from numba import jit, njit


class MSUDenseLeavesDataset(Dataset):

    def __init__(self, filepath, num_targets, random_augmentation=False, augm_probability=0.2):
        self.filepath = filepath
        # each sample is made of image-labels-mask (important to sort by name!)
        self.images = sorted(glob.glob(filepath + '*_img.png'))
        self.labels = sorted(glob.glob(filepath + '*_label.png'))
        self.masks = sorted(glob.glob(filepath + '*_mask.png'))
        self.n_samples = len(self.images)
        self.multiscale_loss_targets = num_targets
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
        image = torch.from_numpy(image.transpose(2, 0, 1) / 255.).float()

        # todo bug in generating dataset made following map have 3 (identical)
        #  channels instead of 1
        label = label[:, :, 0] / 255.
        mask = mask[:, :, 0] / 255.
        # label = torch.from_numpy(label).long()
        # mask = torch.from_numpy(mask).float()

        # todo fix pil transformation
        if self.augmentation:
            # augment data randomly
            if random.random() > 1 - self.probability:
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

        # labels multiscale resizing
        targets, masks = multiscale_target(self.multiscale_loss_targets, label, mask)
        # reverse order of multiscale labels (long cast for CE loss)
        return image, [torch.from_numpy(t).long() for t in reversed(targets)], [torch.from_numpy(m) for m in reversed(masks)]

# normal multiscale does not achieve same results
# def multiscale(n_scaling, target, mask):
#     target = target.astype(np.float32)
#     mask = mask.astype(np.float32)
#     targets = [target]
#     masks = [mask]
#     parent_t = target
#     parent_mask = mask
#     for t in range(n_scaling):
#         scaled_t = cv2.resize(parent_t, (int(parent_t.shape[0]/2), int(parent_t.shape[1]/2)), interpolation=cv2.INTER_NEAREST).astype(np.float32)
#         scaled_m = cv2.resize(parent_mask, (int(parent_t.shape[0]/2), int(parent_t.shape[1]/2)), interpolation=cv2.INTER_NEAREST).astype(np.float32)
#         targets.append(scaled_t)
#         masks.append(scaled_m)
#
#         parent_t = scaled_t
#         parent_mask = scaled_m
#     return targets, masks

@njit
def multiscale_target(n_targets, target, mask):
    # targets = np.empty((self.multiscale_loss_targets, target.shape[0], target.shape[1]))
    targets = []
    targets.append(target.astype(np.float32))
    masks = []
    masks.append(mask.astype(np.float32))
    # outputs as many targets as the number of evaluations in the multiscale loss
    parent_target = target.astype(np.float32).copy() # remember to uniform to same type(float32) or numba will complain
    parent_mask = mask.astype(np.float32).copy()
    for t in range(n_targets - 1):
        scaled_target = np.zeros((int(parent_target.shape[0] / 2), int(parent_target.shape[1] / 2))).astype(np.float32)
        scaled_mask = np.zeros((int(parent_target.shape[0] / 2), int(parent_target.shape[1] / 2))).astype(np.float32)
        for y, i in enumerate(range(1, parent_target.shape[0] - 1, 2)):
            for x, j in enumerate(range(1, parent_target.shape[1] - 1, 2)):
                # check neighbour pixels for edges (clockwise check order)
                if parent_target[i, j - 1] == 1.0 or parent_target[i - 1, j] == 1.0 or \
                        parent_target[i, j + 1] == 1.0 or parent_target[i + 1, j] == 1.0:
                    scaled_target[y, x] = 1.0
                    scaled_mask[y, x] = 1.0
                # else if any of its parents are unknown it is unknown,
                elif parent_target[i, j - 1] == 0.0 or parent_target[i - 1, j] == 0.0 or \
                        parent_target[i, j + 1] == 0.0 or parent_target[i + 1, j] == 0.0:
                    scaled_target[y, x] = 0.0
                    scaled_mask[y, x] = 0.0

                if parent_mask[i, j - 1] == 1.0 or parent_mask[i - 1, j] == 1.0 or \
                        parent_mask[i, j + 1] == 1.0 or parent_mask[i + 1, j] == 1.0:
                    # interior pixel
                    scaled_mask[y, x] = 1.0

        targets.append(scaled_target)
        masks.append(scaled_mask)
        parent_target = scaled_target
        parent_mask = scaled_mask
    return targets, masks


if __name__ == '__main__':
    dataset = MSUDenseLeavesDataset('/home/nick/datasets/DenseLeaves/leaves_edges/', num_targets=5,
                                    random_augmentation=False,
                                    augm_probability=1.0)
    img, l, m = dataset[10]
    print(img.shape)  # , l.shape, m.shape)
    img = img.permute(1, 2, 0).numpy() * 255
    img = img.astype(np.uint8)
    print(img.shape)
    for target, mask in zip(l, m):
        target = target.numpy()
        mask = mask.numpy()
        print(target.shape, mask.shape)
        cv2.imshow('imga', target)
        cv2.imshow('imgb', mask)
        cv2.waitKey(0)
    # cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.imshow('labels', l.numpy().astype(np.uint8)*255)
    # cv2.waitKey(0)
    # cv2.imshow('mask',  m.numpy().astype(np.uint8)*255)
    # cv2.waitKey(0)
