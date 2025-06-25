import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__':
    root = 'D:\\pythonProject\\crowdCounting\\ShanghaiTech'
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    path_sets = [part_A_train, part_A_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        img_path = img_path.replace('\\', '/')
        print(img_path.replace('.jpg', '.csv').replace('images', 'den'))
        img = plt.imread(img_path)
        data = pd.read_csv(img_path.replace('.jpg', '.csv').replace('images', 'den'))
        data_np = data.values
        np.save(img_path.replace('.jpg', '.npy').replace('images', 'ground_truth'), data_np)
