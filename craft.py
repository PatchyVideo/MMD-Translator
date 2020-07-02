
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class CRAFT(nn.Module) :
    def __init__(self) :
        super(CRAFT, self).__init__()
        self.backbone = resnet18(pretrained = True)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
        )

        self.upconv1 = double_conv(0, 512, 192)
        self.upconv2 = double_conv(192, 256, 96)
        self.upconv3 = double_conv(96, 128, 64)
        self.upconv4 = double_conv(64, 64, 32)

    def forward(self, x) :
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x) # 64@128

        h4 = self.backbone.layer1(x) # 64@128
        h8 = self.backbone.layer2(h4) # 128@64
        h16 = self.backbone.layer3(h8) # 256@32
        h32 = self.backbone.layer4(h16) # 512@16
        up32 = F.interpolate(self.upconv1(h32), scale_factor = (2, 2), mode = 'bilinear', align_corners = False) # 256@32
        up16 = F.interpolate(self.upconv2(torch.cat([up32, h16], dim = 1)), scale_factor = (2, 2), mode = 'bilinear', align_corners = False) # 128@64
        up8 = F.interpolate(self.upconv3(torch.cat([up16, h8], dim = 1)), scale_factor = (2, 2), mode = 'bilinear', align_corners = False) # 64@128
        up4 = F.interpolate(self.upconv4(torch.cat([up8, h4], dim = 1)), scale_factor = (2, 2), mode = 'bilinear', align_corners = False) # 64@256
        
        return self.conv_cls(up4)

if __name__ == '__main__' :
    net = CRAFT_net().cuda()
    img = torch.randn(1, 3, 640, 640).cuda()
    print(net(img).shape)
