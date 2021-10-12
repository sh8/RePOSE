from lib.csrc.ransac_voting.ransac_voting_gpu import (
    ransac_voting_layer_v3, estimate_voting_distribution_with_mean)
import torch
from torch import nn
from .resnet import resnet18


class Backbone(nn.Module):
    def __init__(self,
                 ver_dim,
                 seg_dim,
                 fcdim=256,
                 s8dim=128,
                 s4dim=64,
                 s2dim=32,
                 raw_dim=32):
        super(Backbone, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.ver_dim = ver_dim
        self.seg_dim = seg_dim

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim), nn.ReLU(True))
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64

        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim), nn.LeakyReLU(0.1, True))
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim), nn.LeakyReLU(0.1, True))
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim), nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1))

    def forward(self, x):
        x2s, x4s, x8s, x16s, xfc = self.resnet18_8s(x)
        return x2s, x4s, x8s, xfc
