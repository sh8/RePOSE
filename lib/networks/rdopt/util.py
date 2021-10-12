import time

import torch
import torch.nn as nn


def rot_qua_to_mat(quat):
    bs = quat.shape[0]

    norm_quat = torch.norm(quat, dim=1).unsqueeze(1)
    norm_quat = quat / norm_quat

    w, x, y, z = \
        norm_quat[:, 0], norm_quat[:, 1], \
        norm_quat[:, 2], norm_quat[:, 3]

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    mat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                      dim=1).reshape(bs, 3, 3)
    return mat


def rot_mat_to_qua(mat):
    mat[mat != mat] = 0
    w = torch.clamp(1 + mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2], min=1e-6)
    w = torch.sqrt(w) / 2
    assert (w != w).sum() == 0, 'w should be not nan'

    w = w.unsqueeze(1)
    w4 = 4 * w
    x = (mat[:, 2, 1] - mat[:, 1, 2]).unsqueeze(1) / w4
    y = (mat[:, 0, 2] - mat[:, 2, 0]).unsqueeze(1) / w4
    z = (mat[:, 1, 0] - mat[:, 0, 1]).unsqueeze(1) / w4
    qua = torch.cat((w, x, y, z), dim=1)
    return qua

@torch.jit.script
def rot_vec_to_mat(vec):
    bs = vec.shape[0]

    theta = torch.norm(vec, dim=1)
    wx = vec[:, 0] / theta
    wy = vec[:, 1] / theta
    wz = vec[:, 2] / theta

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    R = torch.zeros((bs, 3, 3), device=vec.device)
    R[:, 0, 0] = costheta + wx * wx * (1 - costheta)
    R[:, 1, 0] = wz * sintheta + wx * wy * (1 - costheta)
    R[:, 2, 0] = -wy * sintheta + wx * wz * (1 - costheta)
    R[:, 0, 1] = wx * wy * (1 - costheta) - wz * sintheta
    R[:, 1, 1] = costheta + wy * wy * (1 - costheta)
    R[:, 2, 1] = wx * sintheta + wy * wz * (1 - costheta)
    R[:, 0, 2] = wy * sintheta + wx * wz * (1 - costheta)
    R[:, 1, 2] = -wx * sintheta + wy * wz * (1 - costheta)
    R[:, 2, 2] = costheta + wz * wz * (1 - costheta)

    return R


def rot_mat_to_vec(matrix):
    bs = matrix.shape[0]

    # Axes.
    axis = torch.zeros((bs, 3)).cuda()

    axis[:, 0] = matrix[:, 2, 1] - matrix[:, 1, 2]
    axis[:, 1] = matrix[:, 0, 2] - matrix[:, 2, 0]
    axis[:, 2] = matrix[:, 1, 0] - matrix[:, 0, 1]

    # Angle.
    norm = torch.norm(axis[:, 1:], dim=1).unsqueeze(1)
    r = torch.cat((axis[:, 0].unsqueeze(1), norm), dim=1)
    r = torch.norm(r, dim=1).unsqueeze(1)
    t = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    t = t.unsqueeze(1)
    theta = torch.atan2(r, t - 1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return theta * axis


@torch.jit.script
def spatial_gradient_jit(padded):
    h, w = padded.shape[2:]
    h = h - 2
    w = w - 2
    grad_x = 0.5 * (padded[:, :, 1:h + 1, 2:w + 2] -
                    padded[:, :, 1:h + 1, 0:w])
    grad_y = 0.5 * (padded[:, :, 2:h + 2, 1:w + 1] -
                    padded[:, :, 0:h, 1:w + 1])
    # [bn, height, width, num_channels, 1]
    grad_x = grad_x.permute((0, 2, 3, 1)).unsqueeze(4)
    # [bn, height, width, num_channels, 1]
    grad_y = grad_y.permute((0, 2, 3, 1)).unsqueeze(4)
    grad_xy = torch.cat((grad_x, grad_y), dim=4)
    return grad_xy


def spatial_gradient(inp):
    h, w = inp.shape[2:]
    padded = nn.ZeroPad2d(1)(inp)
    grad_xy = spatial_gradient_jit(padded)
    return grad_xy


@torch.jit.script
def crop_input(inp, bbox):
    bs = inp.shape[0]
    inp_r = []
    for i in range(bs):
        inp_r_ = inp[i, :, bbox[i, 0]:bbox[i, 2], bbox[i, 1]:bbox[i, 3]]
        inp_r.append(inp_r_)
    inp_r = torch.stack(inp_r, dim=0)
    return inp_r


@torch.jit.script
def crop_features(x2s, x4s, x8s, x16s, bbox):
    bs = x2s.shape[0]
    mask_r, x2s_r, x4s_r, x8s_r, x16s_r = [], [], [], [], []
    bbox2s = bbox // 2
    bbox4s = bbox // 4
    bbox8s = bbox // 8
    for i in range(bs):
        x2s_r_ = x2s[i, :, bbox2s[i, 0]:bbox2s[i, 2], bbox2s[i, 1]:bbox2s[i,
                                                                          3]]
        x4s_r_ = x4s[i, :, bbox4s[i, 0]:bbox4s[i, 2], bbox4s[i, 1]:bbox4s[i,
                                                                          3]]
        x8s_r_ = x8s[i, :, bbox8s[i, 0]:bbox8s[i, 2], bbox8s[i, 1]:bbox8s[i,
                                                                          3]]
        x16s_r_ = x16s[i, :, bbox8s[i, 0]:bbox8s[i, 2], bbox8s[i, 1]:bbox8s[i,
                                                                            3]]
        x2s_r.append(x2s_r_)
        x4s_r.append(x4s_r_)
        x8s_r.append(x8s_r_)
        x16s_r.append(x16s_r_)
    x2s_r = torch.stack(x2s_r, dim=0)
    x4s_r = torch.stack(x4s_r, dim=0)
    x8s_r = torch.stack(x8s_r, dim=0)
    x16s_r = torch.stack(x16s_r, dim=0)
    return x2s_r, x4s_r, x8s_r, x16s_r
