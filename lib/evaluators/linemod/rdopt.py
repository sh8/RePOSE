from scipy import spatial
from lib.utils.img_utils import read_depth
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.csrc.nn import nn_utils
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
import os
from lib.utils.linemod import linemod_config
from lib.networks.rdopt.util import rot_vec_to_mat
import torch
import pickle


class Evaluator:
    def __init__(self, result_dir, is_train, dataset=None):
        self.result_dir = result_dir
        if dataset is not None:
            args = DatasetCatalog.get(dataset)
            self.dataset = dataset
        else:
            args = DatasetCatalog.get(cfg.test.dataset)
            self.dataset = cfg.test.dataset
        self.is_train = is_train
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

        cls = cfg.cls_type
        model_path = os.path.join('data/linemod', cls, cls + '.ply')
        self.model = pvnet_data_utils.get_ply_model(model_path)
        self.diameter = linemod_config.diameters[cls] / 100

        self.proj2d = []
        self.add = []
        self.cmd5 = []

    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_targets = pvnet_pose_utils.project(self.model, K,
                                                    pose_targets)
        proj_mean_diff = np.mean(
            np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

        self.proj2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model,
                               pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(
                np.linalg.norm(model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(
                np.linalg.norm(model_pred - model_targets, axis=-1))

        self.add.append(mean_dist < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        translation_distance = np.linalg.norm(pose_pred[:, 3] -
                                              pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def evaluate(self, output, batch):
        R = rot_vec_to_mat(output['R'])[0]
        t = output['t'][0].unsqueeze(1)
        pose_pred = torch.cat((R, t), dim=1)
        pose_pred = pose_pred.detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        K = np.array(anno['K'])
        pose_gt = np.array(anno['pose'])
        if cfg.cls_type in ['eggbox', 'glue']:
            self.add_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt)
        self.projection_2d(pose_pred, pose_gt, K)
        self.cm_degree_5_metric(pose_pred, pose_gt)

    def summarize(self):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('5 cm 5 degree metric: {}'.format(cmd5))

        self.proj2d = []
        self.add = []
        self.cmd5 = []
        return {'proj2d': proj2d, 'add': add, 'cmd5': cmd5}
