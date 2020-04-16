import cv2
import numpy as np
import glob

labels = sorted(glob.glob('/home/nick/datasets/DenseLeaves/leaves_edges/*_label.png'))
masks = sorted(glob.glob('/home/nick/datasets/DenseLeaves/leaves_edges/*_mask.png'))
total = 0
for i, (l, m) in enumerate(zip(labels, masks)):
    li = cv2.imread(l)[:, :, 0]
    mi = cv2.imread(m)[:, :, 0]
    # print(li.shape, mi.shape)
    # count number of edges
    n_edges = np.count_nonzero(li)
    n_pixels = li.shape[0] * li.shape[1]
    n_not_edges = np.count_nonzero(mi) - n_edges
    assert n_not_edges > n_edges
    # print("Edges over interior pixels:", (n_edges/n_not_edges))
    total += (n_edges/n_not_edges)

print("Edges over interior pixels (mean):", (total/i))