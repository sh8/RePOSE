import time
import os

from lib.config import cfg
import neural_renderer as nr
import numpy as np
import torch
from torch import nn

from .backbone import Backbone
from .repose_feat import ReposeFeat
from .gn import GNLayer
from lib.networks.rdopt.texture_net import TextureNet
from lib.csrc.camera_jacobian.camera_jacobian_gpu import calculate_jacobian
from .util import rot_vec_to_mat, spatial_gradient, crop_input, crop_features

GPU_COUNT = torch.cuda.device_count()
MAX_NUM_OF_GN = 5
IN_CHANNELS = 3
OUT_CHANNELS = 3


class RDOPT(nn.Module):
    def __init__(self):
        super(RDOPT, self).__init__()

        filename = f'./data/linemod/{cfg.model}/{cfg.model}.obj'
        vertices, faces, textures = nr.load_obj(filename)

        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])
        self.register_buffer('textures', textures[None, :, :])

        self.renderer = nr.Renderer(image_height=cfg.image_height,
                                    image_width=cfg.image_width,
                                    camera_mode='projection')
        self.texture_net = TextureNet(IN_CHANNELS, OUT_CHANNELS, vertices)
        self.repose_feat = ReposeFeat(OUT_CHANNELS)
        self.gn = GNLayer(OUT_CHANNELS)

    def rend_feature(self, vertices, faces, textures, K, R_vec, t, bbox):
        R_mat = rot_vec_to_mat(R_vec)
        t = t.view(-1, 1, 3)

        f_rend, face_index_map, depth_map = self.renderer(
            vertices, faces, textures, K, R_mat, t, bbox[:, 0:1], bbox[:, 1:2])
        mask = f_rend[:, -1:]
        f_rend = f_rend[:, :-1]

        return f_rend, mask, R_mat, face_index_map, depth_map

    def forward(self, inp, K, x_ini, bbox, x2s, x4s, x8s, xfc):
        output = {}

        bs, _, h, w = inp.shape

        vertices = self.vertices
        faces = self.faces
        # textures = self.textures
        textures = self.texture_net()

        inp = crop_input(inp, bbox)

        x = torch.zeros_like(x_ini)
        x[:, :3] = x_ini[:, :3]
        x[:, 3:] = x_ini[:, 3:]

        if x[0, 5] < 0.5:
            output['R'] = x[:, :3]
            output['t'] = x[:, 3:]
            output['vertices'] = vertices
            return output, 0.0, False
        
        if not self.training:
            x_all = torch.zeros((MAX_NUM_OF_GN + 1, 6), device=x.device)
            x_all[0] = x[0]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        f_inp = self.repose_feat(inp, x2s, x4s, x8s, xfc)

        for i in range(MAX_NUM_OF_GN):
            f_rend, r_mask, R_mat, face_index_map, depth_map = \
                self.rend_feature(vertices, faces, textures,
                                  K, x[:, :3], x[:, 3:], bbox)
            e = f_inp - f_rend
            grad_xy = spatial_gradient(f_rend)
            # Perform anlytical jacobian computation
            J_c = calculate_jacobian(face_index_map, depth_map, K, R_mat,
                                     x[:, :3], x[:, 3:], bbox)
            e = e.permute((0, 2, 3, 1))
            e = e.reshape(bs, -1, 3)
            grad_xy = grad_xy.view(-1, OUT_CHANNELS, 2)
            J_c = J_c.view(-1, 2, 6)
            J = torch.bmm(grad_xy, J_c)
            J = J.reshape(bs, -1, OUT_CHANNELS, 6)
            x = self.gn(x, e, J)
            
            if not self.training:
                x_all[i + 1] = x[0]

        output['R'] = x[:, :3]
        output['t'] = x[:, 3:]
        output['vertices'] = vertices
        
        if not self.training:
            output['R_all'] = x_all[:, :3]
            output['t_all'] = x_all[:, 3:]

        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

        return output, elapsed_time, True


def get_res_rdopt():
    model = RDOPT()
    return model
