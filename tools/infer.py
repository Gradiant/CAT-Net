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
from lib.utils.utils import AverageMeter, create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from pathlib import Path
from project_config import dataset_paths
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import os, sys
from loguru import logger
import mlflow
from PIL import Image

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

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

def _skip_image(label, th):
    logger.info(" *** Size of image {}x{} -> {:.2f}MB".format(label.size()[1],label.size()[2], label.size()[1]*label.size()[2]/1000000))
    mb_size = label.size()[1]*label.size()[2]/1000000
    if mb_size > th:
        return True
    else:
        return False

def _save_onnx_model(model, image, qtable, filename):
    logger.info("... Saving ONNX model")
    dynamic_axes = {'image':{0:'batch_size' , 2:'width', 3:'height'}, 'qtable':{0:'batch_size' , 2:'width', 3:'height'}}
    torch.onnx.export(model, (image, qtable) , filename, opset_version = 13,  input_names = ['image','qtable'], dynamic_axes=dynamic_axes)

def _check_inputs_names(onnx_model):
    inputs = {}
    for inp in onnx_model.graph.input:
        shape = str(inp.type.tensor_type.shape.dim)
        inputs[inp.name] = [int(s) for s in shape.split() if s.isdigit()]
    print(inputs)

def _binarize_prediction(pred, treshold=0.5):
    # CUSTOM THRESHOLD
    pred_bin = np.array(pred)
    pred_bin[pred > treshold] = 1
    pred_bin[pred <= treshold] = 0
    return pred_bin


def _onnx_inference(image, qtable, filename_onnx):
    import onnxruntime as ort
    providers = ['CPUExecutionProvider']
    ort_sess = ort.InferenceSession(filename_onnx, providers=providers)
    inputs_onnx = {'image': image.cpu().detach().numpy(),'qtable':qtable.cpu().detach().numpy()}
    outputs = ort_sess.run(None, inputs_onnx)
    pred_s = np.squeeze(outputs)
    pred = (softmax(pred_s))[1]
    # https://onnxruntime.ai/docs/get-started/with-python.html
    return pred

def _store_maps(filename, pred):
    # filename
    base_name = os.path.splitext(filename)[0]
    base_path = dataset_paths['SAVE_PRED'] / base_name
    filepath_pred =  str(base_path) + "_pred.png"
    filepath = str(base_path) + ".png"      
    logger.info(f"... Saving prediction in {filepath_pred}")
    cv2.imwrite(str(filepath_pred), pred*255.0)

    #plot2
    try:    
        img = cv2.imread( dataset_paths['LOAD_FOLDER']+filename, 1)
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
        logger.info(f"Error occurred while saving output superimpose. ({filename})")

def infer():
    # args = parse_args()
    # Instead of using argparse, force these args:

    ## CHOOSE ##
    IS_SEGMENTATION_MODEL = False
    FULL_OPT = False
    EXPORT_ONNX = True
    COMPARE_ONNX_PTH = True
    SAVE_CSV_RESULTS = False
    SAVE_MAPS = True
    #use cat_2 environment in gea2 to export to ONNX
    onnx_created = False
    show_mlflow = False
    save_metrics = True
    metrics_path = "./data.txt"
    ##working option
    #if FULL_OPT:
    #    args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output_orig/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    #else:
    #    args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output_orig/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    
    if IS_SEGMENTATION_MODEL:
        args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/finetuning_DCT_DOCIMANv1_DOCUMENTS_cm_best.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    else:
        args = argparse.Namespace(cfg='experiments/CAT_DCT_only_cls.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only_cls/best_CAT_DCT_CLS_lr10e4_DOCIMANv1_bs16.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    update_config(config, args)

    # cudnn related setting
    if EXPORT_ONNX:
        import onnx
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
        if IS_SEGMENTATION_MODEL:
            test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # DCT stream
        else:
            test_dataset = splicing_dataset(crop_size=(512,512), grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitraryCls', read_from_jpeg=True)  # DCT stream
  
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
    if IS_SEGMENTATION_MODEL:
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
    else:
        criterion = nn.CrossEntropyLoss()

        model = eval('models.' + config.MODEL.NAME +
                 '.get_cls_net')(config)
 
    

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

    if IS_SEGMENTATION_MODEL:
        avg_acc = AverageMeter()
        avg_p_acc = AverageMeter()
        avg_mIoU = AverageMeter()
        avg_p_mIoU = AverageMeter()
        avg_F1 = AverageMeter()
        avg_p_F1 = AverageMeter()
        avg_AP = AverageMeter()
        avg_p_AP = AverageMeter()
    else:
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
            logger.info("Load input file : {}".format(get_next_filename(index)))
            if 'mask' in get_next_filename(index):
                print("skip mask")
                continue
            
            if IS_SEGMENTATION_MODEL:
                mb_size = label.size()[1]*label.size()[2]/1000000
                if mb_size > 9:
                    print("Skip image, too big image!")
                    continue
            
            size = label.size()
            if not EXPORT_ONNX:
                image = image.cuda()
                label = label.long().cuda()
            
            model.eval()

            # for param in two_inputs_model.parameters():
            #     print(param.data)
            # exit(-1)

            if EXPORT_ONNX:
                if not onnx_created:
                    # two_inputs_model.eval()
                    nameonnx = "CATNET_DCT_CLS.onnx"
                    if IS_SEGMENTATION_MODEL:
                        nameonnx = "CATNET_DCT_only.onnx"
                        
                    _save_onnx_model(two_inputs_model, image, qtable, nameonnx)
                    onnx_created = True
                logger.info("... Loading ONNX model")
                # onnx_model = onnx.load("CATNET_DCT_only.onnx")

                ## print onnx model weights
                # weights = onnx_model.graph.initializer
                # import onnx.numpy_helper as nh
                # np.set_printoptions(threshold=sys.maxsize)
                # print(nh.to_array(weights[0]))
                # exit(-1)

                # print(onnx_model.graph.input[0].type.tensor_type.shape.dim)
                # onnx.checker.check_model(onnx_model)
                # _check_inputs_names(onnx_model)

                pred = _onnx_inference(image, qtable, nameonnx)

                if COMPARE_ONNX_PTH:
                    pred_onnx = pred

            if COMPARE_ONNX_PTH or not EXPORT_ONNX:
            
                _, pred = model(image, label, qtable)
                pred = torch.squeeze(pred, 0)
                pred = F.softmax(pred, dim=0)[1]
                pred = pred.cpu().numpy() 

                if IS_SEGMENTATION_MODEL:
                    im_h = cv2.hconcat([pred, pred_onnx])
                    filename_pred = os.path.splitext(get_next_filename(index))[0] + "_comparemodels.png"
                    cv2.imwrite(str(dataset_paths['SAVE_PRED'] / filename_pred), im_h*255.0)
                    
                    filename_pred = os.path.splitext(get_next_filename(index))[0] + "_difmodels.png"
                    cv2.imwrite(str(dataset_paths['SAVE_PRED'] / filename_pred), (np.abs(pred-pred_onnx))*255.0)
                else:
                    print(f"Checkpoint prediction is {pred} and onnx prediction is {pred_onnx}")

            if SAVE_MAPS and IS_SEGMENTATION_MODEL:
                if EXPORT_ONNX and COMPARE_ONNX_PTH:
                    pred = pred_onnx
                
                pred_bin = _binarize_prediction(pred, treshold=0.5)

                width_im, heigh_im = np.shape(pred)
                logger.info("% mod {:.2f}% and max value {}".format(100*pred_bin.sum()/(width_im*heigh_im), pred.max()))

                if SAVE_CSV_RESULTS:
                    outputfile.write("{},{},{}\n".format(get_next_filename(index), pred.sum()/(width_im*heigh_im), pred.max()))

                _store_maps(get_next_filename(index),  pred)               

            torch.cuda.empty_cache()

    if SAVE_CSV_RESULTS:
        outputfile.close()

if __name__ == '__main__':
    infer()
