import os
import torch
import torch.nn as nn
# import visdom
# import random
from mcnn import MCNN
from dataloader import CrowdDataset
import time

if __name__ == '__main__':
    # vis = visdom.Visdom()
    device = torch.device('cuda')
    mcnn = MCNN().to(device)
    criterion = nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-6, momentum=0.95)

    img_root = 'D:\\pythonProject\\crowdCounting\\ShanghaiTech\\part_A_final\\train_data\\images'
    gt_dmap_root = 'D:\\pythonProject\\crowdCounting\\ShanghaiTech\\part_A_final\\train_data\\ground_truth'
    dataset = CrowdDataset(img_root, gt_dmap_root, 4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    test_img_root = 'D:\\pythonProject\\crowdCounting\\ShanghaiTech\\part_A_final\\test_data\\images'
    test_gt_dmap_root = 'D:\\pythonProject\\crowdCounting\\ShanghaiTech\\part_A_final\\test_data\\ground_truth'
    test_dataset = CrowdDataset(test_img_root, test_gt_dmap_root, 4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    min_mae = 10000
    min_epoch = 0
    train_loss_list = []
    epoch_list = []
    test_error_list = []
    ts = time.time()
    for epoch in range(0, 2000):
        mcnn.train()
        epoch_loss = 0
        for i, (img, gt_dmap) in enumerate(dataloader):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            et_dmap = mcnn(img)
            loss = criterion(et_dmap, gt_dmap)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss:', epoch_loss / len(dataloader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss / len(dataloader))
        torch.save(mcnn.state_dict(), './checkpoints/epoch_' + str(epoch) + '.param')
        mcnn.eval()
        ts_epoch = time.time()
        print('time:', int(ts_epoch - ts), 's')
        ts = ts_epoch

        if epoch % 5 == 0:
            mae = 0
            for i, (img, gt_dmap) in enumerate(test_dataloader):
                img = img.to(device)
                gt_dmap = gt_dmap.to(device)
                et_dmap = mcnn(img)
                mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
                del img, gt_dmap, et_dmap
            if mae / len(test_dataloader) < min_mae:
                min_mae = mae / len(test_dataloader)
                min_epoch = epoch
            test_error_list.append(mae / len(test_dataloader))
            print('epoch:', epoch, 'error:', mae / len(test_dataloader), 'min_mae:', min_mae, 'min_epoch:', min_epoch)

        # vis.line(win=1, X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        # vis.line(win=2, X=epoch_list, Y=test_error_list, opts=dict(title='test_error'))
        # index = random.randint(0, len(test_dataloader) - 1)
        # img, gt_dmap = test_dataset[index]
        # vis.image(win=3, img=img, opts=dict(title='img'))
        # vis.image(win=4, img=gt_dmap / (gt_dmap.max()) * 255, opts=dict(title='gt_dmap(' + str(gt_dmap.sum()) + ')'))
        # img = img.unsqueeze(0).to(device)
        # gt_dmap = gt_dmap.unsqueeze(0)
        # et_dmap = mcnn(img)
        # et_dmap = et_dmap.squeeze(0).detach().cpu().numpy()
        # vis.image(win=5, img=et_dmap / (et_dmap.max()) * 255, opts=dict(title='et_dmap(' + str(et_dmap.sum()) + ')'))
