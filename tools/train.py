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

from stages.experiment.qf_analysis import qf_analysis
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
from lib.core.function import train, validate
from lib.utils.utils import create_logger, FullModel
from lib import models

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
    # args = parse_args()
    # Instead of using argparse, force these args:
    ## CHOOSE ##
    # args = argparse.Namespace(cfg='experiments/CAT_full.yaml', local_rank=0, opts=None)
    args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', local_rank=0, opts=None)
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
                 '.get_seg_model')(config)

    writer_dict = {
        # 'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    if config.DATASET.DATASET == 'splicing_dataset':
        ## CHOOSE ##
        # train_dataset = splicing_dataset(crop_size=crop_size, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), mode='train', DCT_channels=1, read_from_jpeg=True, class_weight=[0.5, 2.5])  # full model
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
    ## CHOOSE ##
    # valid_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), mode="valid", DCT_channels=1, read_from_jpeg=True)  # full model
    valid_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), mode="valid", DCT_channels=1, read_from_jpeg=True)  # only DCT stream

    print(" ***=> DATALOADER val")

    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights).cuda(device=gpus[0])
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights).cuda(device=gpus[0])

    model = FullModel(model, criterion)

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
    best_mIoU, last_epoch = 0, 0

    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        # train
        train_dataset.shuffle()  # for class-balanced sampling
        train(config, epoch, config.TRAIN.END_EPOCH,
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict)

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        # Valid
        if epoch % 1 == 0:
            valid_loss, avg_mIoU, avg_IoU, IoU_array, pixel_acc, mean_acc, confusion_matrix, f1_avg, prec_avg, recall_avg, list_data = \
                validate(config, validloader, model, valid_dataset)

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(3.0)

            if avg_IoU > best_IoU:
                best_IoU = avg_IoU
                torch.save({
                    'epoch': epoch + 1,
                    'best_IoU': avg_IoU,
                    'state_dict': model.model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(final_output_dir, 'best_'+str(epoch)+'_.pth.tar'))
                logger.info("best.pth.tar updated.")

            msg = '(Valid) Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}, avg_mIoU: {: 4.4f}, avg_p_mIoU: {: 4.4f}, Pixel_Acc: {: 4.4f}, Mean_Acc: {: 4.4f}'.format(
                valid_loss, best_IoU, avg_mIoU, avg_IoU, pixel_acc, mean_acc)

            metrics={
                "valid_loss": valid_loss,
                "best_IoU": best_IoU,
                "avg_IoU": avg_IoU,
                "avg__mIoU": avg_mIoU,
                "pixel_acc": pixel_acc,
                "mean_acc": mean_acc,
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
            'best_mIoU': best_mIoU,
            'state_dict': model.model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint_epoch_'+str(epoch+1)+'.pth.tar'))

        output_data = output_folder+"/data_segmentation_"+str(epoch+1)+".txt"
        with open(output_data, "w") as f:
                f.write('\n'.join(list_data)+'\n')

        qf_analysis(output_folder, output_data, epoch=str(epoch+1), mode="seg")


if __name__ == '__main__':
    fire.Fire(train_model)
