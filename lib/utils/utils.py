# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

TH_BIN = 0.65

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels, qtable):
    outputs = self.model(inputs, qtable)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss,0), outputs

class FullModelCombined(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss_seg, loss_cls):
    super(FullModelCombined, self).__init__()
    self.model = model
    self.loss_seg = loss_seg
    self.loss_cls = loss_cls

  def forward(self, inputs, masks, labels, qtable):
    outputs_seg, outputs_cls = self.model(inputs, qtable)
    loss_seg = self.loss_seg(outputs_seg, masks)
    loss_cls = self.loss_cls(outputs_cls, labels)
    loss = loss_seg + loss_cls
    
    return torch.unsqueeze(loss,0), torch.unsqueeze(loss_seg,0), torch.unsqueeze(loss_cls,0), outputs_seg, outputs_cls


windows_mode = True
def get_world_size():
    if windows_mode:
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if windows_mode:
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def _logit_to_prob(pred):
    output = pred
    exp_array = np.exp(output)
    return exp_array / (1+ exp_array)

def _get_prob_forgery_map(pred):
    output_logits = pred.cpu().numpy().transpose(0, 2, 3, 1)
    output = _logit_to_prob(output_logits)
    _,w, h, _ = np.shape(output)
    seg_pred = output[:,:,:,1].reshape((1, w,h))
    return seg_pred


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    seg_pred = _get_prob_forgery_map(pred)

    seg_pred[seg_pred > TH_BIN] = 1
    seg_pred[seg_pred <= TH_BIN] = 0

    seg_label = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_label != ignore
    seg_label = seg_label[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_label * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

def is_class_0(label, size):
    #get data from tensors
    seg_label = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)
    result = True if np.max(seg_label) == 0 else False
    return result    


def get_f1_pixel_level(label, pred, size, num_class, ignore=-1):
    #get data from tensors
    
    seg_pred = _get_prob_forgery_map(pred)
    seg_pred[seg_pred > TH_BIN] = 1
    seg_pred[seg_pred <= TH_BIN] = 0

    seg_label = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)
    label = np.reshape(seg_label, (np.shape(seg_label)[1], np.shape(seg_label)[2]))
    pred = np.reshape(seg_pred, (np.shape(seg_pred)[1], np.shape(seg_pred)[2]))


    if np.max(pred) == np.max(label) and np.max(pred) == 0:
        f1, iou = 1.0, 1.0
        return f1, 0.0, 0.0
    seg_inv, label_inv = np.logical_not(pred), np.logical_not(label)
    true_pos = float(np.logical_and(pred, label).sum())
    false_pos = np.logical_and(pred, label_inv).sum()
    false_neg = np.logical_and(seg_inv, label).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)


    return f1, precision, recall