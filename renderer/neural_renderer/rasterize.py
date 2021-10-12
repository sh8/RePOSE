import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import neural_renderer.cuda.rasterize as rasterize_cuda

DEFAULT_IMAGE_SIZE = 256
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)


class RasterizeFunction(Function):
    '''
    Definition of differentiable rasterize operation
    Some parts of the code are implemented in CUDA
    Currently implemented only for cuda Tensors
    '''
    @staticmethod
    def forward(ctx, faces, textures, image_height, image_width, near, far,
                background_color):
        '''
        Forward pass
        '''

        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.near = near
        ctx.far = far
        ctx.background_color = background_color

        ctx.device = faces.device
        ctx.num_faces = faces.shape[1]
        ctx.batch_size = textures.shape[0]
        ctx.texture_size = textures.shape[3]

        feature_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_height,
                                             ctx.image_width,
                                             ctx.texture_size + 1).fill_(0.0)
        face_index_map = torch.cuda.IntTensor(ctx.batch_size, ctx.image_height,
                                              ctx.image_width).fill_(-1)
        weight_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_height,
                                            ctx.image_width, 3).fill_(0.0)
        depth_map = torch.cuda.FloatTensor(ctx.batch_size, ctx.image_height,
                                           ctx.image_width).fill_(ctx.far)
        lock_map = torch.cuda.IntTensor(ctx.batch_size, ctx.image_height,
                                        ctx.image_width).fill_(0)

        face_index_map, weight_map, depth_map, feature_map = \
            rasterize_cuda.forward(feature_map, face_index_map,
                                   weight_map, depth_map,
                                   lock_map, faces, textures, ctx.image_height,
                                   ctx.image_width, ctx.near, ctx.far)
        ctx.save_for_backward(faces, textures, face_index_map, weight_map)
        ctx.mark_non_differentiable(face_index_map, weight_map)
        return feature_map, face_index_map, depth_map

    @staticmethod
    def backward(ctx, grad_feature_map, grad_face_index_map, grad_depth_map):
        '''
        Backward pass
        '''

        faces, textures, face_index_map, weight_map = ctx.saved_tensors
        # initialize output buffers
        # no need for explicit allocation of cuda.FloatTensor
        # because zeros_like does it automatically
        grad_textures = torch.zeros_like(textures, dtype=torch.float32)

        # get grad_outputs
        grad_feature_map = grad_feature_map.contiguous()

        # backward pass
        grad_textures = rasterize_cuda.backward(face_index_map, weight_map,
                                                grad_feature_map,
                                                grad_textures, ctx.num_faces)

        return None, grad_textures, None, None, None, None, None


class Rasterize(nn.Module):
    '''
    Wrapper around the autograd function RasterizeFunction
    Currently implemented only for cuda Tensors
    '''
    def __init__(self, image_height, image_width, near, far, background_color):
        super(Rasterize, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.near = near
        self.far = far
        self.background_color = background_color

    def forward(self, faces, textures):
        feature, face_index_map, depth_map = RasterizeFunction.apply(
            faces, textures, self.image_height, self.image_width, self.near,
            self.far, self.background_color)
        feature = feature.permute((0, 3, 1, 2))
        return feature, face_index_map, depth_map
