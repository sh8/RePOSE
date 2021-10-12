import torch
from lib.config import cfg, args
import numpy as np
import os


def run_rgb():
    import glob
    from scipy.misc import imread
    import matplotlib.pyplot as plt

    syn_ids = sorted(os.listdir('data/ShapeNet/renders/02958343/'))[-10:]
    for syn_id in syn_ids:
        pkl_paths = glob.glob(
            'data/ShapeNet/renders/02958343/{}/*.pkl'.format(syn_id))
        np.random.shuffle(pkl_paths)
        for pkl_path in pkl_paths:
            img_path = pkl_path.replace('_RT.pkl', '.png')
            img = imread(img_path)
            plt.imshow(img)
            plt.show()


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            network(batch['inp'])
    print(total_time / len(data_loader))


def run_evaluate():
    import time
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from torch.nn import DataParallel

    network = make_network(cfg).cuda()
    nextwork = DataParallel(network)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    outputs = []

    tot_elapsed_time = 0.0
    tot_valid_cnt = 0

    print('Start inference...')
    with torch.inference_mode():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            inp = batch['inp'].cuda()
            K = batch['K'].cuda()
            x_ini = batch['x_ini'].cuda()
            bbox = batch['bbox'].cuda()
            x2s = batch['x2s'].cuda()
            x4s = batch['x4s'].cuda()
            x8s = batch['x8s'].cuda()
            xfc = batch['xfc'].cuda()
            output, elapsed_time, is_valid = network(inp, K, x_ini, bbox, x2s,
                                                     x4s, x8s, xfc)
            if is_valid:
                tot_elapsed_time += elapsed_time
                tot_valid_cnt += 1

            outputs.append(output)

    print('Start computing ADD(-S) metrics...')
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        output = outputs[i]
        evaluator.evaluate(output, batch)

    print('Average Elapsed Time', tot_elapsed_time / tot_valid_cnt)
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        inp = batch['inp'].cuda()
        K = batch['K'].cuda()
        x_ini = batch['x_ini'].cuda()
        bbox = batch['bbox'].cuda()
        x2s = batch['x2s'].cuda()
        x4s = batch['x4s'].cuda()
        x8s = batch['x8s'].cuda()
        xfc = batch['xfc'].cuda()
        with torch.inference_mode():
            output, _, _ = network(inp, K, x_ini, bbox, x2s, x4s, x8s, xfc)
        visualizer.visualize(output, batch, i)


def run_analyze():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.analyzers import make_analyzer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    analyzer = make_analyzer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        analyzer.analyze(output, batch)


def run_linemod():
    from lib.datasets.linemod import linemod_to_coco
    linemod_to_coco.linemod_to_coco(cfg)


if __name__ == '__main__':
    globals()['run_' + args.type]()
