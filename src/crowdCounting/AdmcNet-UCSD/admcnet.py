import torch
import torch.nn as nn


class AdmcNet(nn.Module):
    def __init__(self):
        super(AdmcNet, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 7, padding=3),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 20, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, 5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 14, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]
        self.backend_feat = [256, 128, 64, 32, 16]
        self.frontend = make_layers(self.frontend_feat, in_channels=3)
        self.backend = make_layers(self.backend_feat, in_channels=256, dilation=True)

        self.cbam = CBAM()

        self.output_layer1 = nn.Conv2d(48, 16, kernel_size=3, padding=1)
        self.output_layer2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.output_layer3 = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_mcnn = torch.cat((x1, x2, x3), 1)

        x_csrnet = self.frontend(x)
        x_csrnet = self.backend(x_csrnet)

        x = torch.cat((x_mcnn, x_csrnet), 1)
        x = self.cbam(x)
        x = self.output_layer1(x)
        x = self.output_layer2(x)
        x = self.output_layer3(x)
        return x[0]


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        mix_out = avg_out + max_out
        mix_out = self.sigmoid(mix_out)
        return mix_out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mix_out = torch.cat([avg_out, max_out], dim=1)
        mix_out = self.sigmoid(self.conv1(mix_out))
        return mix_out * x


class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels=48, ratio=16)
        self.spatialattention = SpatialAttention(kernel_size=3)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x


if __name__ == '__main__':
    admcnet = AdmcNet()
    input_img = torch.ones((1, 256, 512, 3))
    out = admcnet(input_img)
    print(out.shape)
