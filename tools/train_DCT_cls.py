import sys, os
from stages.experiment.histogram import show_histogram

from stages.experiment.qf_analysis import qf_analysis
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import logging
import time
import mlflow
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from lib.config import config
from lib.config import update_config
from lib.core.function import train, validate_cls
from lib.utils.utils import create_logger, FullModel

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from stages.experiment.roc_classification import plot_roc_curve


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

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
    
    args = argparse.Namespace(cfg='experiments/CAT_DCT_only_cls.yaml', local_rank=0, opts=None)
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
                 '.get_cls_net')(config)

    writer_dict = {
        # 'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    if config.DATASET.DATASET == 'splicing_dataset':
        train_dataset = splicing_dataset(crop_size=crop_size, grid_crop=True, blocks=('DCTvol', 'qtable'), mode='train', DCT_channels=1, read_from_jpeg=True, class_weight=[0.5, 2.5])  # [0.5, 2.5] only DCT stream
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
    valid_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), mode="valid", DCT_channels=1, read_from_jpeg=True)  # only DCT stream
    logger.info(valid_dataset.get_info())
    print(" ***=> DATALOADER val")

    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # criterion
    criterion = nn.CrossEntropyLoss()

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
    valid_acc = 0
    best_acc = 0
    last_epoch = 0
   
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        # train
        train_dataset.shuffle()  # for class-balanced sampling
        train(config, epoch, config.TRAIN.END_EPOCH,
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict, final_output_dir)

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        # Valid
        valid_loss, valid_acc, list_data = validate_cls(valid_dataset, validloader, model)

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        output_data = "data_"+epoch+".txt"
        with open(output_data, "w") as f:
                f.write('\n'.join(list_data)+'\n')

        qf_analysis(mlflow.get_artifact_uri()[7:], output_data, cls_mode=True, epoch=str(epoch))
        show_histogram(mlflow.get_artifact_uri()[7:], output_data, str(epoch))
        plot_roc_curve(mlflow.get_artifact_uri()[7:], output_data, str(epoch))

        if valid_acc > best_acc:
            best_acc = valid_acc

            torch.save({
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'state_dict': model.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'best.pth.tar'))
            logger.info("best.pth.tar updated.")

        msg = '(Valid) Loss: {:.3f}, Val Acc: {: 4.4f}'.format(valid_loss, valid_acc)

        metrics={
                "valid_loss": valid_loss,
                "valid_acc": valid_acc
        }

        mlflow.log_metrics(metrics)

        logging.info(msg)


        logger.info('=> saving checkpoint to {}'.format(
            os.path.join(final_output_dir, 'checkpoint.pth.tar')))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))


    
if __name__ == '__main__':
    train_model()
