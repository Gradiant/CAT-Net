# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
"""
Modified by Myung-Joon Kwon
mjkwon2021@gmail.com
July 14, 2020
"""

import sys, os
from stages.experiment.histogram import show_histogram

from stages.experiment.qf_analysis import qf_analysis
from stages.experiment.roc_classification import plot_roc_curve
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import logging
import time
import timeit
import mlflow
import gc
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim

from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import validate_combined, train_combined
from lib.utils.utils import create_logger, FullModelCombined
from lib import models
import torch.nn as nn

from Splicing.data.data_core import SplicingDataset as splicing_dataset
import fire

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def train_model():
    args = argparse.Namespace(cfg='experiments/CAT_DCT_only_combined.yaml', local_rank=0, opts=None)
    output_folder = mlflow.get_artifact_uri()[7:]
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    gpus = list(config.GPUS)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    gpus[0] = 0

    # build model
    model = eval('models.' + config.MODEL.NAME +
                 '.get_combined_model')(config)

    writer_dict = {
        # 'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    if config.DATASET.DATASET == 'splicing_dataset':
        train_dataset = splicing_dataset(crop_size=crop_size, grid_crop=True, blocks=('DCTvol', 'qtable'), mode='train', DCT_channels=1, read_from_jpeg=True, class_weight=[0.5, 2.5])  # only DCT stream
        logger.info(train_dataset.get_info())
    else:
        raise ValueError("Not supported dataset type.")

    print(" ***=> DATALOADER train")
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*1,#1 instead of len(gpus)
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False, )

    # validation
    valid_dataset = splicing_dataset(crop_size=crop_size, grid_crop=True, blocks=('DCTvol', 'qtable'), mode="valid", DCT_channels=1, read_from_jpeg=True)  # only DCT stream

    print(" ***=> DATALOADER val")

    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion_seg = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights).cuda(device=gpus[0])
    else:
        criterion_seg = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights).cuda(device=gpus[0])

    criterion_cls = nn.CrossEntropyLoss()
    model = FullModelCombined(model, criterion_seg, criterion_cls)

    # optimizer
    logger.info(f"# params with requires_grad = {len([c for c in model.parameters() if c.requires_grad])}, "
                f"# params freezed = {len([c for c in model.parameters() if not c.requires_grad])}")
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                          filter(lambda p: p.requires_grad,
                                                 model.parameters()),
                                      'lr': config.TRAIN.LR}],
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(train_dataset.__len__() /
                         config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_IoU, best_accuracy, last_epoch = 0, 0, 0

    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        # train
        train_dataset.shuffle()  # for class-balanced sampling
        train_combined(config, epoch, config.TRAIN.END_EPOCH,
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict)

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        # Valid
        if epoch % 1 == 0:
            valid_acc, valid_loss, valid_loss_seg, valid_loss_cls, avg_IoU, IoU_array, avg_mIoU, pixel_acc, confusion_matrix, f1_avg, prec_avg, recall_avg, list_data = \
                validate_combined(config, validloader, model, valid_dataset)

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(3.0)

            if avg_IoU > best_IoU and valid_acc > best_accuracy:
                best_IoU = avg_IoU
                best_accuracy = valid_acc
                torch.save({
                    'epoch': epoch + 1,
                    'best_IoU': best_IoU,
                    'best_accuracy': best_accuracy,
                    'state_dict': model.model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(final_output_dir, 'best_'+str(epoch+1)+'.pth.tar'))
                logger.info("best.pth.tar updated.")

            msg = '(Valid) Accuracy: {:.3f}, (Valid) Loss: {:.3f}, (Valid) Loss Segmentation: {:.3f}, (Valid) Loss Classification: {:.3f}, avg_IoU: {: 4.4f}, Best_IoU: {: 4.4f}, avg_mIoU: {: 4.4f}, Pixel_Acc: {: 4.4f}'.format(
                valid_acc, valid_loss, valid_loss_seg, valid_loss_cls, avg_IoU, best_IoU, avg_mIoU, pixel_acc)

            metrics={
                "classification_acc": valid_acc,
                "valid_loss": valid_loss,
                "avg_IoU": avg_IoU,
                "best_IoU": best_IoU,
                "avg_mIoU": avg_mIoU,
                "pixel_acc": pixel_acc,
                "iou_class0": IoU_array[0],
                "iou_class1": IoU_array[1],
                "f1": f1_avg,
                "precission": prec_avg,
                "recall": recall_avg
            }

            mlflow.log_metrics(metrics)

            logging.info(msg)
            logging.info("IOU class : {}".format(IoU_array))
            logging.info("confusion_matrix:")
            logging.info(confusion_matrix)
            logging.info("-------------------")
            logging.info("F1 total avg = {:.3f}".format(f1_avg))
            logging.info("Prec total avg = {:.3f}".format(prec_avg))
            logging.info("Recall total avg = {:.3f}".format(recall_avg))


        else:
            logging.info("Skip validation.")

        logger.info('=> saving checkpoint to {}'.format(
            os.path.join(final_output_dir, 'checkpoint_epoch_'+str(epoch+1)+'.pth.tar')))
        torch.save({
            'epoch': epoch + 1,
            'best_IoU': best_IoU,
            'state_dict': model.model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint_epoch_'+str(epoch+1)+'.pth.tar'))

        output_data = output_folder+"/data_combined_"+str(epoch+1)+".txt"
        with open(output_data, "w") as f:
                f.write('\n'.join(list_data)+'\n')

        show_histogram(output_folder, output_data, str(epoch+1), mode="combined")
        plot_roc_curve(output_folder, output_data, str(epoch+1), mode="combined")
        qf_analysis(output_folder, output_data, epoch=str(epoch+1), mode="combined")
        

if __name__ == '__main__':
    fire.Fire(train_model)
