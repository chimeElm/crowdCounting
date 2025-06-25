from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image


class CrowdDataset(Dataset):
    def __init__(self, img_root, gt_dmap_root, scale_factor=16):
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.img_names = [filename for filename in os.listdir(img_root) if
                          os.path.isfile(os.path.join(img_root, filename))]
        self.img_names.sort(key=lambda x: eval(x.split('.')[0]))
        self.n_samples = len(self.img_names)
        self.scale_factor = scale_factor

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = Image.open(os.path.join(self.img_root, img_name))
        gt_dmap = np.load(os.path.join(self.gt_dmap_root, img_name.replace('.jpg', '.npy')))

        width, height = img.size
        new_width = (width // self.scale_factor) * self.scale_factor // 2
        new_height = (height // self.scale_factor) * self.scale_factor // 2
        img = img.resize((new_width, new_height))
        gt_dmap = Image.fromarray(gt_dmap)
        gt_dmap = gt_dmap.resize((img.width, img.height))
        gt_dmap = np.array(gt_dmap)
        gt_dmap *= 4
        img = np.array(img)
        if len(img.shape) == 2:  # expand grayscale image to three channel.
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), 2)

        img_tensor = torch.tensor(img, dtype=torch.float)
        gt_dmap_tensor = torch.tensor(gt_dmap, dtype=torch.float)
        return img_tensor, gt_dmap_tensor


if __name__ == '__main__':
    dataset = CrowdDataset(
        'D:\\pythonProject\\crowdCounting\\ShanghaiTech\\part_A_final\\test_data\\images',
        'D:\\pythonProject\\crowdCounting\\ShanghaiTech\\part_A_final\\test_data\\ground_truth'
    )
    print(dataset[0][0].shape)
