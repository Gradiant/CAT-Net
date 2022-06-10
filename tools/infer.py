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
import cv2

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
from loguru import logger


def softmax(x):
    
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

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
    EXPORT_ONNX = False
    SAVE_CSV_RESULTS = False
    #use cat_2 environment in gea2 to export to ONNX
    onnx_created = False

    ##working option
    if FULL_OPT:
        args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output_orig/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    else:
        args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output_orig/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])

    update_config(config, args)

    # cudnn related setting
    if EXPORT_ONNX:
        import onnx
        import onnxruntime as ort   
        cudnn.benchmark = False
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = False
    else:
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
    if EXPORT_ONNX:
        if config.LOSS.USE_OHEM:
            criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=test_dataset.class_weights)

        else:
            criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=test_dataset.class_weights)
    else:
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
    logger.info('=> loading model from {}'.format(model_state_file))

    two_inputs_model = model
    model = FullModel(model, criterion)

    map_location = 'cpu' if EXPORT_ONNX else 'cuda:0'
    checkpoint = torch.load(model_state_file, map_location=map_location)

    model.model.load_state_dict(checkpoint['state_dict'])


    if not EXPORT_ONNX:
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

    if SAVE_CSV_RESULTS:
        outputfile = open("catnetresultscopymove.csv","w")
        outputfile.write("filename,sum,max\n")
        

    with torch.no_grad():
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):
            logger.info("Load input file : {}".format(get_next_filename(index)))
            if 'mask' in get_next_filename(index):
                print("skip mask")
                continue
            logger.info(" *** Size of image {}x{} -> {:.2f}MB".format(label.size()[1],label.size()[2], label.size()[1]*label.size()[2]/1000000))
            logger.info(" *** Shape of input {}".format(image.size()))

            mb_size = label.size()[1]*label.size()[2]/1000000
            if mb_size > 8.8:
                logger.info(" *** Skip image, too big image!")
                continue
            
            size = label.size()
            if not EXPORT_ONNX:
                image = image.cuda()
                label = label.long().cuda()
            
            model.eval()

            if EXPORT_ONNX:
                if not onnx_created:
                    logger.info("... Saving ONNX model")
                    torch.onnx.export(two_inputs_model, (image, qtable) , "CATNET_DCT_only.onnx", opset_version = 14,  input_names = ['image','qtable'])
                    onnx_created = True
                logger.info("... Loading ONNX model")
                onnx_model = onnx.load("CATNET_DCT_only.onnx")
                # onnx.checker.check_model(onnx_model)

                inputs = {}
                for inp in onnx_model.graph.input:
                    shape = str(inp.type.tensor_type.shape.dim)
                    inputs[inp.name] = [int(s) for s in shape.split() if s.isdigit()]

                providers = [
                    'CPUExecutionProvider',
                ]
                ort_sess = ort.InferenceSession('CATNET_DCT_only.onnx',providers=providers)
                inputs_onnx = {'image': image.cpu().detach().numpy(),'qtable':qtable.cpu().detach().numpy()}
                # inputs_onnx = {'image': image.cpu().detach().numpy(),'label': label.cpu().detach().numpy(),'qtable':qtable.cpu().detach().numpy()}
                outputs = ort_sess.run(None, inputs_onnx)
                pred_s = np.squeeze(outputs)
                pred = (softmax(pred_s.T).T)[1]
                # https://onnxruntime.ai/docs/get-started/with-python.html

            else:
                _, pred = model(image, label, qtable)

                print(pred.size())
                pred = torch.squeeze(pred, 0)

                pred = F.softmax(pred, dim=0)[1]
                pred = pred.cpu().numpy() 

            # CUSTOM THRESHOLD
            # pred = F.upsample(input=pred, size=(size[-2], size[-1]), mode='bilinear')
            # output_logits = pred.cpu().numpy().transpose(0, 2, 3, 1)
            # exp_array = np.exp(output_logits)
            # output =  exp_array / (1+ exp_array)
            # _,w, h, _ = np.shape(output)

            TH_BIN = 0.5
            pred_bin = np.array(pred)
            pred_bin[pred > TH_BIN] = 1
            pred_bin[pred <= TH_BIN] = 0
            #########################

            width_im, heigh_im = np.shape(pred)
            logger.info("% mod {:.2f}% and max value {}".format(100*pred_bin.sum()/(width_im*heigh_im), pred.max()))

            if SAVE_CSV_RESULTS:
                outputfile.write("{},{},{}\n".format(get_next_filename(index), pred.sum()/(width_im*heigh_im), pred.max()))

            # filename
            filename = os.path.splitext(get_next_filename(index))[0] + ".png"
            filename_pred = os.path.splitext(get_next_filename(index))[0] + "_pred.png"

            filepath_pred = dataset_paths['SAVE_PRED'] / filename_pred
            filepath = dataset_paths['SAVE_PRED'] / filename
            
            print(f"... Saving prediction in {filepath_pred}")
            cv2.imwrite(str(filepath_pred), pred*255.0)
            #plot2
            try:
                
                img = cv2.imread( dataset_paths['LOAD_FOLDER']+get_next_filename(index), 1)
                if len(np.shape(img))>2:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = img
                    color_img = cv2.cvtColor(img, cv2.CV_GRAY2RGB)
                    img = color_img
                h, w = np.shape(pred)

                heatmap_img = cv2.applyColorMap(np.uint8(pred*255.0), cv2.COLORMAP_JET)
                img_resized = cv2.resize(img, (w,h), interpolation = cv2.INTER_CUBIC)
                fin = cv2.addWeighted(heatmap_img, 0.5, img_resized, 0.5, 0)
                fin_resized = cv2.resize(fin, (w*6,h*6), interpolation = cv2.INTER_CUBIC)
                
                cv2.imwrite(str(filepath), fin_resized)
            except:
                logger.info(f"Error occurred while saving output superimpose. ({get_next_filename(index)})")

            torch.cuda.empty_cache()

    if SAVE_CSV_RESULTS:
        outputfile.close()

if __name__ == '__main__':
    main()
