from __future__ import division
import math

import torch
import torch.nn as nn
import numpy

import neural_renderer as nr


class Renderer(nn.Module):
    def __init__(self,
                 image_height=256,
                 image_width=256,
                 background_color=[0, 0, 0],
                 fill_back=True,
                 camera_mode='projection',
                 K=None,
                 R=None,
                 t=None,
                 perspective=True,
                 viewing_angle=30,
                 near=0.1,
                 far=100):
        super(Renderer, self).__init__()
        # rendering
        self.image_height = image_height
        self.image_width = image_width
        self.background_color = background_color
        self.fill_back = fill_back
        self.rasterize = nr.Rasterize(image_height, image_width, near, far,
                                      background_color)

        # camera
        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.K = K
            self.R = R
            self.t = t
            if isinstance(self.K, numpy.ndarray):
                self.K = torch.cuda.FloatTensor(self.K)
            if isinstance(self.R, numpy.ndarray):
                self.R = torch.cuda.FloatTensor(self.R)
            if isinstance(self.t, numpy.ndarray):
                self.t = torch.cuda.FloatTensor(self.t)
        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [
                0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)
            ]
            self.camera_direction = [0, 0, 1]
        else:
            raise ValueError(
                'Camera mode has to be one of projection, look or look_at')

        self.near = near
        self.far = far

    def forward(self,
                vertices,
                faces,
                textures=None,
                K=None,
                R=None,
                t=None,
                offset_y=None,
                offset_x=None,
                down_sampling=1):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility
        with the Chainer implementation
        '''

        return self.render(vertices, faces, textures, K, R, t, offset_y,
                           offset_x, down_sampling)

    def render(self,
               vertices,
               faces,
               textures,
               K=None,
               R=None,
               t=None,
               offset_y=None,
               offset_x=None,
               down_sampling=1):

        K = K / down_sampling
        K[:, 2, 2] = 1

        image_height = self.image_height // down_sampling
        image_width = self.image_width // down_sampling

        vertices = nr.projection(vertices, K, R, t, image_height, image_width,
                                 offset_y, offset_x)

        face_vertices = nr.vertices_to_faces(vertices, faces)
        face_textures = nr.vertices_to_faces(textures, faces)

        out, face_index_map, depth_map = self.rasterize(
            face_vertices, face_textures)

        return out, face_index_map, depth_map
