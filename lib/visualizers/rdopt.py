import os
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
from lib.networks.rdopt.util import rot_vec_to_mat
from matplotlib import cm

mean = pvnet_config.mean
std = pvnet_config.std

viridis = cm.get_cmap('viridis', 256)


class Visualizer:
    def __init__(self, split):
        if split == 'test':
            args = DatasetCatalog.get(cfg.test.dataset)
        else:
            args = DatasetCatalog.get(cfg.train.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch, step):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean,
                                        std).permute(1, 2, 0)
        # mask = output['mask'][0].detach().cpu().numpy()

        R = rot_vec_to_mat(output['R_all']).detach().cpu().numpy()
        t = output['t_all'].detach().cpu().numpy()[:, :, None]
        pose_preds = np.concatenate((R, t), axis=2)

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        pose_gt = np.array(anno['pose'])
        K = np.array(anno['K'])

        cond1 = np.linalg.norm(pose_preds[-1, :, 3] - pose_preds[0, :, 3]) > 0.03
        cond2 = np.linalg.norm(pose_preds[-1, :, 3] - pose_gt[:, 3]) < 0.01
        if not (cond1 and cond2):
            return
        print('Passed!!', step)

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        _, ax = plt.subplots(1)
        ax.imshow(inp)

        for i, pose_pred in enumerate(pose_preds):
            color = viridis((i + 1) / 7)
            corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
            ax.add_patch(
                patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]],
                                fill=False,
                                linewidth=1,
                                edgecolor=color))
            ax.add_patch(
                patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]],
                                fill=False,
                                linewidth=1,
                                edgecolor=color))

        os.makedirs(f'linemod/samples/{cfg.test.dataset}/{cfg.cls_type}', exist_ok=True)

        plt.savefig(f'linemod/samples/{cfg.test.dataset}/{cfg.cls_type}/{cfg.cls_type}_{step}.pdf')
        plt.clf()

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.add_patch(
            patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]],
                            fill=False,
                            linewidth=1,
                            edgecolor='g'))
        ax.add_patch(
            patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]],
                            fill=False,
                            linewidth=1,
                            edgecolor='g'))
        plt.savefig(f'linemod/samples/{cfg.test.dataset}/{cfg.cls_type}/{cfg.cls_type}_{step}_gt.pdf')
        plt.clf()

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        plt.savefig(f'linemod/samples/{cfg.test.dataset}/{cfg.cls_type}/{cfg.cls_type}_{step}_orig.pdf')

    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean,
                                        std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['center'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['center_2d'])
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        # plt.savefig('test.jpg')
        plt.show()
        # plt.close(0)
