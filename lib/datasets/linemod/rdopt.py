import os
import random
import math

import cv2
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
from scipy import ndimage
from lib.utils.linemod import linemod_config
from lib.utils.pvnet import (pvnet_data_utils, pvnet_pose_utils,
                             visualize_utils)
from lib.config import cfg

from transforms3d.euler import mat2euler, euler2mat


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split, transforms=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        cgmodel_dir = os.path.join(cfg['cgmodel_dir'], cfg.model,
                                   f'{cfg.model}.ply')
        self.model = pvnet_data_utils.get_ply_model(cgmodel_dir)

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfg

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]
        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        path = path.replace('benchwise', 'benchvise')
        mask_path = anno['mask_path'].replace('benchwise', 'benchvise')
        inp = Image.open(path)
        w, h = inp.size
        K = np.array(anno['K']).astype(np.float32)
        pose = np.array(anno['pose']).astype(np.float32)
        R = pose[:, :3]
        R = cv2.Rodrigues(R)[0].reshape(3)
        t = np.expand_dims(pose[:, 3], axis=1)
        cls = anno['cls']
        if cls == 'benchwise':
            cls = 'benchvise'
        cls_idx = linemod_config.linemod_cls_names.index(cls) + 1
        mask = pvnet_data_utils.read_linemod_mask(mask_path, anno['type'],
                                                  cls_idx)

        if 'occlusion' in self.data_root:
            dataset = 'LinemodOccTest'
        else:
            dataset = cfg.train.dataset if self.split == 'train' else cfg.test.dataset
        directory = f'cache/{dataset}/{cfg.model}'

        filename = f'{directory}/{img_id}.npy'
        result = np.load(filename, allow_pickle=True).item()
        bbox = result['bbox']
        x_ini = result['x_ini']

        filename = f'{directory}/{img_id}_features.npz'
        features = np.load(filename, allow_pickle=True)

        inp = np.asarray(inp).astype(np.uint8)
        mask = np.asarray(mask).astype(np.uint8)

        return inp, K, R, t, x_ini, bbox, features

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, K, R, t, x_ini, bbox, features = self.read_data(img_id)

        if self._transforms is not None:
            img = self._transforms(img)

        ret = {
            'inp': img,
            'K': K,
            'x_ini': x_ini,
            'bbox': bbox,
            'R': R,
            't': t,
            'img_id': img_id,
            'x2s': features['x2s'][0],
            'x4s': features['x4s'][0],
            'x8s': features['x8s'][0],
            'xfc': features['xfc'][0],
        }

        return ret

    def __len__(self):
        return len(self.img_ids)
