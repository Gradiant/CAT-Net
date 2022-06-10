"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 June 7, 2021
"""
import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import shutil

import logging
import time
import timeit
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from pathlib import Path
from project_config import dataset_paths
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    # args = parse_args()
    # Instead of using argparse, force these args:

    ## CHOOSE ##
    FULL_OPT = False
    ##working option
    # if FULL_OPT:
    #     args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    # else:    
    #     args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/checkpoint.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    ## CHOOSE ##
    if FULL_OPT:
        test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # full model
    else:
        test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # DCT stream

    print(test_dataset.get_info())

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # must be 1 to handle arbitrary input sizes
        shuffle=False,  # must be False to get accurate filename
        num_workers=1,
        pin_memory=False)
    
    gpus = list(config.GPUS)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    gpus[0] = 0

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=test_dataset.class_weights).cuda()

    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=test_dataset.class_weights).cuda(device=gpus[0])

    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        raise ValueError("Model file is not specified.")
    print('=> loading model from {}'.format(model_state_file))

    model = FullModel(model, criterion)
    checkpoint = torch.load(model_state_file, map_location='cuda:0')

    model.model.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    dataset_paths['SAVE_PRED'].mkdir(parents=True, exist_ok=True)


    def get_next_filename(i):
        dataset_list = test_dataset.dataset_list
        it = 0
        while True:
            if i >= len(dataset_list[it]):
                i -= len(dataset_list[it])
                it += 1
                continue
            name = dataset_list[it].get_tamp_name(i)
            name = os.path.split(name)[-1]
            return name

    outputfile = open("catnetresultscopymove.csv","w")
    outputfile.write("filename,sum,max\n")

    with torch.no_grad():
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):
            print("load file : {}".format(get_next_filename(index)))
            if 'mask' in get_next_filename(index):
                print("skip mask")
                continue
            print("Size of image {}x{} -> {:.2f}MB".format(label.size()[1],label.size()[2], label.size()[1]*label.size()[2]/1000000))
            
            mb_size = label.size()[1]*label.size()[2]/1000000
            if mb_size > 8.3:
                print("Skip image, too big image!")
                continue
            
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            model.eval()
            _, pred = model(image, label, qtable)
            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()

            width_im, heigh_im = np.shape(pred)
            print("% mod {:.2f}% and max value {}".format(100*pred.sum()/(width_im*heigh_im), pred.max()))

            outputfile.write("{},{},{}\n".format(get_next_filename(index), pred.sum()/(width_im*heigh_im), pred.max()))

            # filename
            filename = os.path.splitext(get_next_filename(index))[0] + ".png"
            filepath = dataset_paths['SAVE_PRED'] / filename

            #plot2
            # try:
            #     import cv2
            #     print(get_next_filename(index))
            #     img = cv2.imread('/home/dperez/workspace/bbdd/ariadnext/GRADIANT_EVALUATION/'+get_next_filename(index), 1)
            #     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     h, w = np.shape(pred)

            #     heatmap_img = cv2.applyColorMap(np.uint8(pred*255.0), cv2.COLORMAP_JET)
            #     img_resized = cv2.resize(img, (w,h), interpolation = cv2.INTER_CUBIC)

            #     fin = cv2.addWeighted(heatmap_img, 0.5, img_resized, 0.5, 0)
            #     cv2.imwrite(str(filepath), fin)
            # except:
            #     print(f"Error occurred while saving output superimpose. ({get_next_filename(index)})")

            del image
            del label
            torch.cuda.empty_cache()

            # plot
            try:
                width = pred.shape[1]  # in pixels
                fig = plt.figure(frameon=False)
                dpi = 40  # fig.dpi
                fig.set_size_inches(width / dpi, ((width * pred.shape[0])/pred.shape[1]) / dpi)
                sns.heatmap(pred, vmin=0, vmax=1, cbar=False, cmap='jet', )
                plt.axis('off')
                plt.savefig(filepath, bbox_inches='tight', transparent=True, pad_inches=0)
                plt.close(fig)
            except:
                print(f"Error occurred while saving output. ({get_next_filename(index)})")

    outputfile.close()

if __name__ == '__main__':
    main()
