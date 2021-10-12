from torch.autograd import Function

import lib.csrc.camera_jacobian.camera_jacobian as camera_jacobian


class CameraJacobian(Function):
    '''
    Calculate Camera Jacobian
    '''
    @staticmethod
    def forward(ctx, face_index_map, depth_map, K, R, w, t, bbox):
        '''
        Forward pass
        '''

        jacobian_c = camera_jacobian.forward(face_index_map, depth_map, K, R,
                                             w, t, bbox)
        return jacobian_c

    @staticmethod
    def backward(ctx, grad_jacobian_c):
        '''
        Backward pass
        '''

        return None, None, None, None, None, None, None


def calculate_jacobian(face_index_map, depth_map, K, R, w, t, bbox):
    jacobian_c = CameraJacobian.apply(face_index_map, depth_map, K, R, w, t,
                                      bbox)
    return jacobian_c
