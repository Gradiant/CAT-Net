"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 June 7, 2021
"""
import sys, os

from sklearn.metrics import average_precision_score
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
from lib.utils.utils import AverageMeter, create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from pathlib import Path
from project_config import dataset_paths
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import os
import mlflow
from PIL import Image

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


def infer():
    # args = parse_args()
    # Instead of using argparse, force these args:

    ## CHOOSE ##
    FULL_OPT = False
    show_mlflow = False
    save_metrics = True
    metrics_path = "./data.txt"
    ##working option
    # if FULL_OPT:
    #     args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    # else:    
    #     args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])

    args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/finetuning_DCT_DOCIMANv1_DOCUMENTS_cm_best.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
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
        num_workers=0,
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

    avg_acc = AverageMeter()
    avg_p_acc = AverageMeter()
    avg_mIoU = AverageMeter()
    avg_p_mIoU = AverageMeter()
    avg_F1 = AverageMeter()
    avg_p_F1 = AverageMeter()
    avg_AP = AverageMeter()
    avg_p_AP = AverageMeter()
    list_data = []

    with torch.no_grad():
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):
            print("load file : {}".format(get_next_filename(index)))
            if 'mask' in get_next_filename(index):
                print("skip mask")
                continue
            print("Size of image {}x{} -> {:.2f}MB".format(label.size()[1],label.size()[2], label.size()[1]*label.size()[2]/1000000))
            
            mb_size = label.size()[1]*label.size()[2]/1000000
            if mb_size > 9:
                print("Skip image, too big image!")
                continue
            
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            model.eval()
            _, pred = model(image, label, qtable)
            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred1 = pred.cpu().numpy()

            pred = pred.unsqueeze(0).unsqueeze(0)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), mode='bilinear', align_corners=False)

            pred_r = pred.cpu().numpy().squeeze(axis=0).squeeze(axis=0).ravel()
            label_r = label.cpu().numpy().squeeze(axis=0).ravel()
            pred_r = pred_r[label_r != -1]
            label_r = label_r[label_r != -1]
            bin_pred = (pred_r >= 0.5).astype(np.float)
            correct = (bin_pred == label_r).astype(np.float)
            incorrect = (bin_pred != label_r).astype(np.float)
            TP = np.count_nonzero(correct[label_r==1])
            TN = np.count_nonzero(correct[label_r==0])
            FP = np.count_nonzero(incorrect[label_r==0])
            FN = np.count_nonzero(incorrect[label_r==1])

            mean_IoU = 0.5 * (TP / np.maximum(1.0, TP + FP + FN)) + 0.5 * (TN / np.maximum(1.0, FP + TN + FN))
            avg_mIoU.update(mean_IoU)
            p_mIoU = 0.5 * (FN / np.maximum(1.0, FN + TP + TN)) + 0.5 * (FP / np.maximum(1.0, FP + TP + TN))
            avg_p_mIoU.update(np.maximum(mean_IoU, p_mIoU))
            acc = (TP+TN)/(TP+TN+FP+FN)
            avg_acc.update(acc)
            p_acc = np.maximum(acc, (FP+FN)/(TP+TN+FP+FN))
            avg_p_acc.update(p_acc)
            F1 = (2*TP) / np.maximum(1.0, 2*TP+FN+FP)
            avg_F1.update(F1)
            p_f1 = (2*FN) / np.maximum(1.0, 2*FN+TP+TN)
            avg_p_F1.update(np.maximum(F1, p_f1))

            AP = average_precision_score(label_r, pred_r)
            avg_AP.update(AP)
            p_AP = average_precision_score(label_r, 1-pred_r)
            avg_p_AP.update(np.maximum(AP, p_AP))

            width_im, heigh_im = np.shape(pred1)
            print("% mod {:.2f}% and max value {}".format(100*pred1.sum()/(width_im*heigh_im), pred1.max()))
            
            print(mean_IoU, p_mIoU, p_AP, p_f1)
            qf1 = get_next_filename(index).split("_")[-2]
            qf2 = get_next_filename(index).split("_")[-1].split(".")[0]
            print(get_next_filename(index), qf1, qf2)
            list_data.append(','.join((get_next_filename(index), qf1, qf2, str(mean_IoU), str(p_mIoU), str(p_AP), str(p_f1), str(pred1.max()) )))

            # filename
            filename = os.path.splitext(get_next_filename(index))[0] + ".png"
            if show_mlflow:
                Path(mlflow.get_artifact_uri()[7:]+"/Predictions/").mkdir(parents=True, exist_ok=True)
                filepath = mlflow.get_artifact_uri()[7:]+"/Predictions/"+filename
            else:
                filepath = dataset_paths['SAVE_PRED'] / filename
            

            del image
            del label
            del pred
            torch.cuda.empty_cache()

            # plot
            try:
                width = pred1.shape[1]  # in pixels
                fig = plt.figure(frameon=False)
                dpi = 40  # fig.dpi
                fig.set_size_inches(width / dpi, ((width * pred1.shape[0])/pred1.shape[1]) / dpi)
                sns.heatmap(pred1, vmin=0, vmax=1, cbar=False, cmap='jet', )
                plt.axis('off')
                plt.savefig(filepath, bbox_inches='tight', transparent=True, pad_inches=0)
                plt.close(fig)
            except:
                print(f"Error occurred while saving output. ({get_next_filename(index)})")
    
    if save_metrics:
        with open(metrics_path, "w") as f:
            f.write('\n'.join(list_data)+'\n')

    results = {'avg_p_acc': avg_p_acc.average(),
               'avg_mIoU': avg_mIoU.average(),
               'avg_p_mIoU': avg_p_mIoU.average(),
               'avg_p_F1': avg_p_F1.average(),
               'avg_p_AP': avg_p_AP.average(),
               'avg_AP':  avg_AP.average(),
               'avg_p_AP': avg_p_AP.average()
               }
    print(results)

    if show_mlflow:
        mlflow.log_metrics(results)
    

if __name__ == '__main__':
    infer()
