"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 June 7, 2021
"""
import json
import sys, os

from sklearn.metrics import average_precision_score
from project_config import project_root
from stages.experiment.histogram import show_histogram
from stages.experiment.roc_classification import plot_roc_curve

from stages.experiment.qf_analysis import qf_analysis
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
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
from lib.core.function import get_next_filename, validate
from lib.utils.utils import AverageMeter, AverageMeter, FullModelCombined

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from pathlib import Path
from project_config import dataset_paths
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import os
import mlflow
from PIL import Image
import mlflow
from PIL import Image
import numpy as np
import fire
import torch.nn as nn


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


def infer_combined(show_mlflow=False):

    ## CHOOSE ##
    FULL_OPT = False
    save_metrics = True
    if show_mlflow:
        metrics_path = mlflow.get_artifact_uri()[7:]
        Path(mlflow.get_artifact_uri()[7:]+"/Masks/").mkdir(parents=True, exist_ok=True)
    else:
        metrics_path = project_root

    args = argparse.Namespace(cfg='experiments/CAT_DCT_only_combined.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only_combined/checkpoint_epoch_1.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

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
        criterion_seg = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=test_dataset.class_weights).cuda()

    else:
        criterion_seg = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=test_dataset.class_weights).cuda(device=gpus[0])

    model = eval('models.' + config.MODEL.NAME +
                 '.get_combined_model')(config)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        raise ValueError("Model file is not specified.")
    print('=> loading model from {}'.format(model_state_file))

    criterion_cls = nn.CrossEntropyLoss()
    model = FullModelCombined(model, criterion_seg, criterion_cls)
    checkpoint = torch.load(model_state_file, map_location='cuda:0')

    model.model.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    dataset_paths['SAVE_PRED'].mkdir(parents=True, exist_ok=True)

    avg_acc = AverageMeter()
    avg_p_acc = AverageMeter()
    avg_mIoU = AverageMeter()
    avg_IoU = AverageMeter()
    avg_p_mIoU = AverageMeter()
    avg_F1 = AverageMeter()
    avg_p_F1 = AverageMeter()
    avg_AP = AverageMeter()
    avg_p_AP = AverageMeter()
    list_data = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    len_testset = 0
    len_single = 0
    len_double = 0
    correct_cls = 0
    correct_single = 0
    correct_double = 0
    list_data = []
    val_accuracy = 0
                
    with torch.no_grad():
        for index, (image, mask, label, qtable) in enumerate(tqdm(testloader)):
            filename = get_next_filename(index, test_dataset)
            print("load file : {}".format(filename))
            if 'mask' in filename:
                print("skip mask")
                continue
            print("Size of image {}x{} -> {:.2f}MB".format(mask.size()[1],mask.size()[2], mask.size()[1]*mask.size()[2]/1000000))
            
            mb_size = mask.size()[1]*mask.size()[2]/1000000
            if mb_size > 12.2:
                print("Skip image, too big image!")
                continue
            
            if show_mlflow:
                mask_np = mask.cpu().numpy()
                Image.fromarray((mask_np[0] * 255).astype(np.uint8)).save(metrics_path+"/Masks/"+filename.split(".")[-2]+".png")
            size = mask.size()
            image = image.cuda()
            mask = mask.long().cuda()
            label = label.long().cuda()
            model.eval()

            if index >= 10:
                starter.record()
                _, _, _, pred_seg, pred_cls = model(image, mask, label, qtable)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)
            else:
                _, _, _, pred_seg, pred_cls = model(image, mask, label, qtable)
            
            pred_seg = torch.squeeze(pred_seg, 0)
            pred_seg = F.softmax(pred_seg, dim=0)[1]
            pred1_seg = pred_seg.cpu().numpy()

            _, pred_class = torch.max(pred_cls, 1)
            pred_cls = torch.squeeze(pred_cls, 0)
            pred_cls = F.softmax(pred_cls, dim=0)[1]
            pred_cls = pred_cls.cpu().numpy()


            print("CLS pred is {}".format(pred_cls))
            if int(pred_class) == int(label):
                correct_cls += 1
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
                val_accuracy = (100 * correct_cls) / (len_testset)
                print(filename, int(label), pred_cls)
                print("Accuracy: ",val_accuracy)
                if len_single > 0:
                    print("Accuracy single: ",(100 * correct_single) / (len_single))
                if len_double > 1:
                    print("Accuracy double: ",(100 * correct_double) / (len_double))
               
            pred_seg = pred_seg.unsqueeze(0).unsqueeze(0)
            if pred_seg.size()[-2] != size[-2] or pred_seg.size()[-1] != size[-1]:
                pred_seg = F.upsample(pred_seg, (size[-2], size[-1]), mode='bilinear', align_corners=False)

            pred_seg = pred_seg.cpu().numpy().squeeze(axis=0).squeeze(axis=0)
            mask = mask.cpu().numpy().squeeze(axis=0)

            if show_mlflow:
                Path(metrics_path+"/Predictions/").mkdir(parents=True, exist_ok=True)
                filepath = metrics_path+"/Predictions/"+filename
            else:
                filepath = dataset_paths['SAVE_PRED'] / filename
            
            pred_mask = (pred_seg >= 0.5)
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(filepath)

            pred_r = pred_seg.ravel()
            label_r = mask.ravel()
            pred_r = pred_r[label_r != -1]
            label_r = label_r[label_r != -1]
            bin_pred = (pred_r >= 0.5).astype(np.float)
            correct = (bin_pred == label_r).astype(np.float)
            incorrect = (bin_pred != label_r).astype(np.float)
            TP = np.count_nonzero(correct[label_r==1])
            TN = np.count_nonzero(correct[label_r==0])
            FP = np.count_nonzero(incorrect[label_r==0])
            FN = np.count_nonzero(incorrect[label_r==1])
            IoU = TP / np.maximum(1.0,(TP+FP+FN))
            avg_IoU.update(IoU)
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

            width_im, heigh_im = np.shape(pred1_seg)
            print("% mod {:.2f}% and max value {}".format(100*pred1_seg.sum()/(width_im*heigh_im), pred1_seg.max()))
            
            print(IoU, mean_IoU, p_AP, p_f1)
            
            filename = os.path.splitext(filename)[0] + ".png"
            if show_mlflow:
                Path(mlflow.get_artifact_uri()[7:]+"/Predictions/").mkdir(parents=True, exist_ok=True)
                filepath = mlflow.get_artifact_uri()[7:]+"/Predictions/"+filename
            else:
                filepath = dataset_paths['SAVE_PRED'] / filename
    
            list_data.append(','.join((filename, str(IoU), str(int(label)), str(pred_cls), str(int(pred_class)))))

            del image
            del label
            del mask
            torch.cuda.empty_cache()

    results = {'single_acc': (100 * correct_single) / (len_single),
               'double_acc': (100 * correct_double) / (len_double),
               'avg_classification_acc': val_accuracy,
               'avg_p_acc': avg_p_acc.average(),
               'avg_mIoU': avg_mIoU.average(),
               'avg_IoU': avg_IoU.average(),
               'avg_p_F1': avg_p_F1.average(),
               'avg_p_AP': avg_p_AP.average(),
               'avg_AP':  avg_AP.average(),
               'avg_p_AP': avg_p_AP.average()
               }
    print(results)

    if show_mlflow:
        mlflow.log_metrics(results)
        
    if save_metrics:
        output_data = metrics_path+"/data_combined.txt"
        with open(output_data, "w") as f:
            f.write('\n'.join(list_data)+'\n')
        show_histogram(metrics_path, output_data, epoch=None, mode="combined")
        plot_roc_curve(metrics_path, output_data, epoch=None, mode="combined")
        qf_analysis(metrics_path, output_data, epoch=None, mode="combined")

    mean_syn = np.sum(np.array(timings)) / len(timings)
    std_syn = np.std(np.array(timings))
    print(mean_syn, std_syn)
    with open(str(metrics_path)+"/inference_time.json", "w") as f:
        json.dump(
            {
                "inference_time": [
                    {
                        "mean": np.mean(mean_syn),
                        "std": np.max(std_syn),
                    }
                ],
            },
            f,
            indent=4,
        )

    
    
if __name__ == '__main__':
    fire.Fire(infer_combined)
