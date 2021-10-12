"""
Example 5. Rendering an image
"""
import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


class Model(nn.Module):
    def __init__(self, filename_obj):
        super(Model, self).__init__()
        # load .obj
        vertices, faces, textures = nr.load_obj(filename_obj)
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])
        self.register_buffer('textures', textures[None, :, :])

        # setup renderer
        renderer = nr.Renderer(image_height=480,
                               image_width=640,
                               camera_mode='projection')

        K = torch.Tensor([[572.4114, 0.0000, 148.2611],
                          [0.0000, 573.5704, 459.0490],
                          [0.0000, 0.0000, 1.0000]])
        R = torch.Tensor([[-0.8033, 0.2854,
                           -0.4163], [0.4520, 0.8342, -0.3860],
                          [0.2247, -0.5214, -0.8232]])
        t = torch.Tensor([-0.0823, -0.12, 1.2])
        self.register_buffer('K', K[None, :, :])
        self.register_buffer('R', R[None, :, :])
        self.register_buffer('t', t[None, None, :])

        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, self.textures, self.K,
                              self.R, self.t)
        return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io',
                        '--filename_obj',
                        type=str,
                        default=os.path.join(data_dir, 'cat.obj'))
    args = parser.parse_args()

    model = Model(args.filename_obj)
    model.cuda()

    images = model()

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        images = model()
    print(prof)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    COUNT = 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(COUNT):
        images = model()
    end.record()
    torch.cuda.synchronize()
    print('Elaseped time on rendering:', start.elapsed_time(end) / COUNT)

    image = images.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0
    image = image[256:(256 + 192), 56:(56 + 192)]
    image = Image.fromarray(image.astype(np.uint8))
    image.save('./examples/data/example5.png')


if __name__ == '__main__':
    main()
