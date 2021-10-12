from scipy import spatial
import torch.nn as nn
import torch
from lib.config import cfg
from lib.networks.rdopt.util import rot_vec_to_mat


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

    def forward(self, batch):
        output = self.net(batch['inp'], batch['mask'], batch['K'],
                          batch['x_ini'], batch['bbox'], batch['kpt_3d'],
                          batch['R'], batch['t'], batch['use_random'])

        scalar_stats = {}
        loss = 0

        bs = batch['inp'].shape[0]
        if self.training:
            gt_mask = output['gt_mask']
            f_inp = output['f_inp'] * gt_mask
            f_rend = output['f_rend'] * gt_mask
            diff_loss = (f_inp - f_rend)**2
            diff_loss = diff_loss.view(bs, -1)
            gt_mask = gt_mask.view(bs, -1)
            gt_mask_sum = gt_mask.sum(dim=1)
            diff_loss = diff_loss.sum(dim=1)
            diff_loss = diff_loss / (gt_mask_sum + 1e-10)
            scalar_stats.update({'diff_loss': diff_loss.mean()})
            loss += 0.1 * diff_loss.sum()

        vertices = output['vertices']
        R_gt = batch['R']
        t_gt = batch['t'].view(-1, 1, 3)
        Rm_gt = rot_vec_to_mat(R_gt).transpose(2, 1)
        v_gt = torch.add(torch.bmm(vertices, Rm_gt), t_gt)

        R_ini = output['R_ini']
        t_ini = output['t_ini'].view(-1, 1, 3)
        Rm_ini = rot_vec_to_mat(R_ini).transpose(2, 1)
        v_ini = torch.add(torch.bmm(vertices, Rm_ini), t_ini)

        R = output['R']
        t = output['t'].view(-1, 1, 3)
        Rm = rot_vec_to_mat(R).transpose(2, 1)
        v = torch.add(torch.bmm(vertices, Rm), t)

        # v_ini = v_ini.view(-1, 3)
        # v_gt = v_gt.view(-1, 3)

        if cfg.cls_type not in ['eggbox', 'glue']:
            pose_ini_loss = torch.norm(v_ini - v_gt, 2, -1).mean(dim=1)
            pose_loss = torch.norm(v - v_gt, 2, -1).mean(dim=1)
        else:
            pose_ini_cdist = torch.cdist(v_ini, v_gt, 2)
            pose_cdist = torch.cdist(v, v_gt, 2)
            pose_ini_loss = torch.min(pose_ini_cdist, dim=1)[0].mean(dim=1)
            pose_loss = torch.min(pose_cdist, dim=1)[0].mean(dim=1)

        scalar_stats.update({'pose_ini_loss': pose_ini_loss.mean()})
        scalar_stats.update({'pose_loss': pose_loss.mean()})

        pose_loss = pose_loss.sum()
        loss += pose_loss

        scalar_stats.update({'loss': loss.mean()})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
