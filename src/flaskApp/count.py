import torch
import matplotlib.pyplot as plt
from admcnet import AdmcNet
from PIL import Image
import numpy as np

model_param_path = 'epoch.param'
device = torch.device('cuda')
admcnet = AdmcNet().to(device)
admcnet.load_state_dict(torch.load(model_param_path))
admcnet.eval()


def estimate_density_map(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    img = torch.tensor(img, dtype=torch.float)
    img = img.to(device)
    img = img.unsqueeze(0).permute(0, 1, 2, 3)
    et_dmap = admcnet(img).detach()
    num = int(et_dmap.data.sum().item())
    et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
    plt.imshow(et_dmap)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(img_path.replace('img', 'res').replace('temp', 'teme'), bbox_inches='tight', pad_inches=0)
    plt.close()
    print(img_path, img.shape, num)
    return num


if __name__ == '__main__':
    print(estimate_density_map('static/img/0.jpg'))
