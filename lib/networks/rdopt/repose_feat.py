import torch
from torch import nn

from lib.config import cfg


class ReposeFeat(nn.Module):
    def __init__(self,
                 feat_dim,
                 s16dim=256,
                 s8dim=128,
                 s4dim=64,
                 s2dim=32,
                 raw_dim=32):

        super(ReposeFeat, self).__init__()

        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + s16dim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up8sto4s = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim), nn.LeakyReLU(0.1, True))
        self.up4sto2s = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim), nn.LeakyReLU(0.1, True))
        self.up2storaw = nn.Upsample(scale_factor=2,
                                     mode='bilinear',
                                     align_corners=cfg.align_corners)

        self.convraw = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, feat_dim, 1, 1))

    def forward(self, x, x2s, x4s, x8s, xfc):
        f8s = self.conv8s(torch.cat([xfc, x8s], 1))
        f8s = self.up8sto4s(f8s)
        f4s = self.conv4s(torch.cat([f8s, x4s], 1))
        f2s = self.up4sto2s(f4s)
        f2s = self.conv2s(torch.cat([f2s, x2s], 1))
        fraws = self.up2storaw(f2s)
        x = self.convraw(torch.cat([fraws, x], 1))
        return x
