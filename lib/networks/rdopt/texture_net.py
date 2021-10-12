import torch
import torch.nn as nn


class TextureNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, vertices):
        super(TextureNet, self).__init__()

        num_vertices = vertices.shape[0]
        textures = torch.empty(1, num_vertices, in_channels)
        self.fc = nn.Sequential(nn.Linear(in_channels, 32), nn.ReLU(),
                                nn.Linear(32, out_channels))
        self.textures = nn.Parameter(textures)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.textures)

    def forward(self):
        textures = self.fc(self.textures)
        return textures
