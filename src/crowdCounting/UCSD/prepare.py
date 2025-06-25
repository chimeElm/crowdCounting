import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt


def gaussian_filter_density():
    img_shape = [img.shape[0], img.shape[1]]
    print('Shape of current image:', img_shape, 'totally need generate', len(points), 'gaussian kernels')
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density
    leafsize = 512
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    distances, locations = tree.query(points, k=4)
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


if __name__ == '__main__':
    root = 'D:/pythonProject/crowdCounting/UCSD'
    part_A_train = os.path.join(root, 'train_data', 'images')
    part_A_test = os.path.join(root, 'test_data', 'images')
    path_sets = [part_A_train, part_A_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.png')):
            img_paths.append(img_path)
    for img_path in img_paths:
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        points = np.load(img_path.replace('.png', '.npy').replace('images', 'annotations'))
        den = gaussian_filter_density()
        np.save(img_path.replace('.png', '.npy').replace('images', 'ground_truth'), den)
