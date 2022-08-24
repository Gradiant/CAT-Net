"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 June 7, 2021
"""
import sys, os
from tkinter import Label
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
    parser = argparse.ArgumentParser(description='Infere classification network')

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
    args = argparse.Namespace(cfg='experiments/CAT_DCT_only_cls.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only_cls/best_CAT_DCT_CLS_lr10e4_DOCIMANv1_bs16.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    test_dataset = splicing_dataset(crop_size=(512,512), grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitraryCls', read_from_jpeg=True)  # DCT stream

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
    criterion = nn.CrossEntropyLoss()

    model = eval('models.' + config.MODEL.NAME +
                 '.get_cls_net')(config)

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

    len_testset = 0
    len_single = 0
    len_double = 0
    valid_acc = 0
    correct = 0
    correct_single = 0
    correct_double = 0
    single_acc = 0
    double_acc = 0
    list_data = []
    with torch.no_grad():
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):

            image = image.cuda()
            label = label.long().cuda()
            model.eval()
            _, output = model(image, label, qtable)

            filename = os.path.splitext(get_next_filename(index))[0] + ".png"
            _, pred1 = torch.max(output, 1)
            pred = torch.squeeze(output, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()

            print("CLS pred is {}".format(pred))
            if int(pred1) == int(label):
                correct += 1
                if int(label) == 0:
                    correct_single += 1
                else:
                    correct_double += 1
            len_testset = index + 1
            if int(label) == 0:
                len_single += 1
            if int(label) == 1:
                len_double += 1
            
            if index % 1 == 0:
                print(filename, int(label), int(pred1), pred)
                print("Accuracy: ",(100 * correct) / (len_testset))
                if len_single > 0:
                    print("Accuracy single: ",(100 * correct_single) / (len_single))
                if len_double > 1:
                    print("Accuracy double: ",(100 * correct_double) / (len_double))
            
            list_data.append(','.join((filename, str(int(label)), str(pred))))

            del image
            del label
            torch.cuda.empty_cache()

    with open("data_cls.txt", "w") as f:
        f.write('\n'.join(list_data)+'\n')

    valid_acc = 100 * correct / len_testset
    single_acc = 100 * correct_single / len_single
    double_acc = 100 * correct_double / len_double
    print("Test accuracy:", valid_acc)
    print("Single accuracy:", single_acc)
    print("Double accuracy:", double_acc)
    outputfile.close()

if __name__ == '__main__':
    main()
