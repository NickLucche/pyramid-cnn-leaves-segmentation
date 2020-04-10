import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = '/home/nick/datasets/DenseLeaves/train/leaf00001_000_img.png'
filename = '/home/nick/datasets/DenseLeaves/train/leaf00001_000_seg.png'
image = cv2.imread(filename)
print(image.shape)
print(np.unique(image))
for color in np.unique(image):
    im = image.copy()
    im[im!=color] = 0
    im[im==color] = 255
    plt.imshow(im)
    plt.show()
# im = cv2.Canny(image, 50, 150)
# cv2.imshow('window', image)
# cv2.waitKey(0)
# plt.imshow(im)
# plt.show()
# plt.imshow(image[:, :, 1])
# plt.show()
# plt.imshow(image[:, :, 2])
# plt.show()
# image[image>0]=255
# image = np.sum(image, axis=2)
# image[image>255] = 255
# print(image.shape, np.unique(image))
# plt.imshow(image, cmap='gray')
# plt.show()