from argparse import ArgumentParser

import cv2
import numpy as np
import sys
import os
from numba import jit
import glob
import tqdm


def generate_patches_from_image(image, segm_map, mask, patch_size, filename, threshold=100):
    """
    Returns a set of cropped patched from the provided image
    by tiling over the image with a window of size patch_size.
    Only patches containing at least 'threshold' labeled pixels
    will be returned.
    """
    patch_no = 0
    filename = filename[:-8]  # remove "_seg.png"
    for y in range(patch_size[0], image.shape[0], patch_size[0]):
        for x in range(patch_size[1], image.shape[1], patch_size[1]):
            # keep it simple, avoid padding
            mask_patch = mask[y - patch_size[0]:y, x - patch_size[1]:x]
            if np.count_nonzero(mask_patch) > threshold:
                # write to disk image-labels-mask
                image_patch = image[y - patch_size[0]:y, x - patch_size[1]:x, :]
                label_patch = segm_map[y - patch_size[0]:y, x - patch_size[1]:x]
                # print(f'Writing to: {filename}_{patch_no}_img.png')
                # print(image_patch.shape, mask_patch.shape, label_patch.shape)
                cv2.imwrite(f'{filename}_{patch_no}_img.png', image_patch)
                cv2.imwrite(f'{filename}_{patch_no}_mask.png', (mask_patch * 255).astype(np.uint8))
                cv2.imwrite(f'{filename}_{patch_no}_label.png', (label_patch * 255).astype(np.uint8))
                patch_no += 1


# generate binary edges and mask from instance segmentation
@jit(nopython=True)
def generate_edges_and_mask(image):
    # weird way to process it but this is how it was intended (took a while to figure it out..)
    segm = np.empty((image.shape[0], image.shape[1])).astype(np.int32)
    segm = image[:, :, 0] + 256 * image[:, :, 1] + 256 * 256 * image[:, :, 2]
    new_segm = np.zeros(segm.shape)
    mask = np.zeros(segm.shape)
    for i in range(1, segm.shape[0] - 1):
        for j in range(1, segm.shape[1] - 1):
            # spot contours
            if segm[i - 1, j] != segm[i + 1, j] or segm[i, j - 1] != segm[i, j + 1] or \
                    segm[i + 1, j - 1] != segm[i + 1, j + 1] or segm[i - 1, j - 1] != segm[i - 1, j + 1]:
                new_segm[i, j] = 1.0
                mask[i, j] = 1.0
            if segm[i, j] != 0:
                mask[i, j] = 1.0
    return new_segm, mask


if __name__ == '__main__':
    # go through dataset and generate binary edge leaves from instance segmentation labels
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Path to the original dataset (avoid final slash / in name)', required=True, type=str)
    parser.add_argument('-p', '--patch-size', type=tuple, default=(128, 128), help='Size of the cropped patches (default: 128x128)')
    parser.add_argument('-s', '--subdir', type=str, default='train',
                        help='Subdirectory to process, assuming original dataset organization', choices=['train', 'val', 'test'])
    parser.add_argument('-o', '--output-dir', help='Where to write new dataset to')
    args = parser.parse_args()
    if not len(sys.argv) > 1:
        print(
            f"Usage: {sys.argv[0]} -d dataset_directory_root (expecting train/val/test inside) [patch_size] [subdir_to_process] [output_directory]")
        exit(-1)

    dirname = 'leaves_edges'
    if not args.output_dir and not os.path.exists(os.path.join(args.dataset, dirname)):
        original_umask = os.umask(0)
        os.makedirs(os.path.join(args.dataset, dirname + '/'), original_umask)

    # assume ~1/4 of image has to be labeled for patch threshold (in case of 128x128 patch size)
    threshold = 3800
    print(args.subdir, args.dataset)
    # loop through instances BUT ONLY NON-PREVIOUSLY ROTATED ONES
    for filename in tqdm.tqdm(glob.glob(args.dataset + f'/{args.subdir}/*000_seg.png')):
        # print(filename, dataset, dirname, os.path.join(dataset, dirname))
        label = cv2.imread(filename)
        original_image = cv2.imread(filename.replace('seg', 'img'))
        # cv2.imshow('a',label)
        # cv2.waitKey(0)
        # cv2.imshow('b',original_image)
        # cv2.waitKey(0)
        # exit(0)
        # generate edge labels and mask from instance segmentation image
        edges_map, mask = generate_edges_and_mask(label)

        f = filename.split('/')[-1]
        # now split image in multiple patches (raw image augmentation)

        output_dir = os.path.join(args.dataset, dirname, f) if not args.output_dir else os.path.join(args.output_dir, f)
        generate_patches_from_image(original_image, edges_map, mask, args.patch_size, output_dir,
                                    threshold=threshold)

    # ax[2].imshow(cv2.imread(filenameim) / 255. + mask[:, :, None])
