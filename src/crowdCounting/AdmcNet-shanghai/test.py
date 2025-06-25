import torch
import matplotlib.pyplot as plt
from admcnet import AdmcNet
from dataloader import CrowdDataset


def cal_mae():
    device = torch.device('cuda')
    admcnet = AdmcNet().to(device)
    admcnet.load_state_dict(torch.load(model_param_path))
    dataset = CrowdDataset(img_root, gt_dmap_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    admcnet.eval()
    mae = 0
    with torch.no_grad():
        for i, (img, gt_dmap) in enumerate(dataloader):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            et_dmap = admcnet(img)
            mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
            print(et_dmap.data.sum().item(), gt_dmap.data.sum().item(), '>>>')
            del img, gt_dmap, et_dmap
    print('model_param_path:', model_param_path, 'MAE:', mae / len(dataloader))


def estimate_density_map(index):
    device = torch.device('cuda')
    admcnet = AdmcNet().to(device)
    admcnet.load_state_dict(torch.load(model_param_path))
    dataset = CrowdDataset(img_root, gt_dmap_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    admcnet.eval()
    for i, (img, gt_dmap) in enumerate(dataloader):
        if i == index:
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            et_dmap = admcnet(img).detach()
            gt_dmap = gt_dmap.squeeze(0).squeeze(0).cpu().numpy()
            et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(img.shape, gt_dmap.shape, et_dmap.shape, '>>>')
            plt.imshow(et_dmap)
            plt.savefig('density_map_et.png')
            plt.imshow(gt_dmap)
            plt.savefig('density_map_gt.png')
            break


if __name__ == '__main__':
    img_root = 'D:\\pythonProject\\crowdCounting\\ShanghaiTech\\part_A_final\\test_data\\images'
    gt_dmap_root = 'D:\\pythonProject\\crowdCounting\\ShanghaiTech\\part_A_final\\test_data\\ground_truth'
    model_param_path = 'D:\\pythonProject\\crowdCounting\\AdmcNet-shanghai\\checkpoints\\epoch_360.param'
    # cal_mae()
    estimate_density_map(7)
