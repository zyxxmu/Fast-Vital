import os
import json
import numpy as np


def gen_config(args):

    if args.seq != '':
        # generate config from a sequence name

        seq_home = '/home/zyx/Binary-Vital/datasets/TEST'
        result_home = './results'

        seq_name = args.seq
        #img_dir = os.path.join(seq_home, seq_name, 'img')
        img_dir = os.path.join(seq_home, seq_name)
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]
        
        with open(gt_path) as f:
            gt = np.loadtxt((x.replace('\t',',') for x in f), delimiter=',')

        if gt.shape[1] == 8:
            x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

        init_bbox = gt[0]

        result_dir = os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        savefig_dir = os.path.join(result_dir, 'figs')
        result_path = os.path.join(result_dir, 'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None

    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path