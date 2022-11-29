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

import os
from pathlib import Path
import time
import mlflow

import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn import functional as F

from lib.utils.utils import AverageMeter
from lib.utils.utils import get_confusion_matrix
from lib.utils.utils import get_f1_pixel_level
from lib.utils.utils import is_class_0
from lib.utils.utils import adjust_learning_rate
from lib.utils.utils import get_world_size, get_rank
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from PIL import Image

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    # writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    world_size = get_world_size()

    for i_iter, (images, labels, qtable) in enumerate(trainloader):
        images = images.cuda()
        labels = labels.long().cuda()
        losses, _ = model(images, labels, qtable)  # _ : output of the model (see utils.py)
        loss = losses.mean()
        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)
        
        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            import sys
            sys.stdout.write('\r'+str(msg))

            mlflow.log_metrics({"loss":print_loss, "lr": lr})
            
            global_steps += 1
            writer_dict['train_global_steps'] = global_steps

def train_combined(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict):
    
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_loss_seg = AverageMeter()
    ave_loss_cls = AverageMeter()

    tic = time.time()
    cur_iters = epoch*epoch_iters
    global_steps = writer_dict['train_global_steps']
    world_size = get_world_size()

    
    for i_iter, (images, masks, labels, qtable) in enumerate(trainloader):
        images = images.cuda()
        masks = masks.long().cuda()
        labels = labels.long().cuda()
        size = masks[0].size()
        losses, losses_seg, losses_cls, pred_seg, pred_cls = model(images, masks, labels, qtable)  # _ : output of the model (see utils.py)

        loss = losses.mean()
        losses_seg = losses_seg.mean()
        losses_cls = losses_cls.mean()

        reduced_loss = reduce_tensor(loss)

        reduced_loss_seg = reduce_tensor(losses_seg)
        reduced_loss_cls = reduce_tensor(losses_cls)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        ave_loss_seg.update(reduced_loss_seg.item())
        ave_loss_cls.update(reduced_loss_cls.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)
        
        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average() / world_size
            print_loss_seg = ave_loss_seg.average() / world_size
            print_loss_cls = ave_loss_cls.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}, Classification loss: {:.6f}, Segmentation loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss, print_loss_cls, print_loss_seg)
            import sys
            sys.stdout.write('\r'+str(msg))

            mlflow.log_metrics({"loss":print_loss, "seg_loss":print_loss_seg, "cls_loss":print_loss_cls, "lr": lr})
            
            global_steps += 1
            writer_dict['train_global_steps'] = global_steps


def validate(config, testloader, model, test_dataset):
    
    world_size = get_world_size()
    model.eval()
    
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    avg_mIoU = AverageMeter()
    avg_p_mIoU = AverageMeter()
    f1_array = np.array([])
    precission_array = np.array([])
    recall_array = np.array([])
    
    list_data = []
    ave_loss = AverageMeter()
    avg_mIoU = AverageMeter()
    avg_IoU = AverageMeter()
    precission = 0
    f1_image = 0
    Path(mlflow.get_artifact_uri()[7:]+"/Masks/").mkdir(parents=True, exist_ok=True)
    Path(mlflow.get_artifact_uri()[7:]+"/Predictions/").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):
            filename = get_next_filename(index, test_dataset)
            Image.fromarray((label.cpu().numpy()[0] * 255).astype(np.uint8)).save(mlflow.get_artifact_uri()[7:]+"/Masks/"+filename.split(".")[-2]+".png")
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            
            losses, pred = model(image, label, qtable)
            pred1 = torch.squeeze(pred, 0)
            pred1 = F.softmax(pred1, dim=0)[1]
            pred1 = pred1.cpu().numpy()
            
            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')

            pred1_seg = torch.squeeze(pred, 0)
            pred1_seg = F.softmax(pred1_seg, dim=0)[1]
            pred1_seg = pred1_seg.unsqueeze(0).unsqueeze(0)
            if pred1_seg.size()[-2] != size[-2] or pred1_seg.size()[-1] != size[-1]:
                pred1_seg = F.upsample(pred1_seg, (size[-2], size[-1]), mode='bilinear', align_corners=False)

            pred1_seg = pred1_seg.cpu().numpy().squeeze(axis=0).squeeze(axis=0)
            pred_mask = (pred1_seg >= 0.5)
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mlflow.get_artifact_uri()[7:]+"/Predictions/"+filename)


            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            if not(is_class_0(label, size)):
                f1_image, precission, recall = get_f1_pixel_level(
                label, 
                pred, 
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
                
                f1_array = np.append(f1_array, f1_image)
                precission_array = np.append(precission_array, precission)
                recall_array = np.append(recall_array,recall)
                #class 0 always returns 0 for all metrics

            current_confusion_matrix = get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            confusion_matrix += current_confusion_matrix
            # mIoU
            pos = current_confusion_matrix.sum(1)  # ground truth label count
            res = current_confusion_matrix.sum(0)  # prediction count
            tp = np.diag(current_confusion_matrix)  # Intersection part
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))  # Union part
            mean_IoU = IoU_array.mean()
            avg_mIoU.update(mean_IoU)
            TN = current_confusion_matrix[0, 0]
            FN = current_confusion_matrix[1, 0]
            FP = current_confusion_matrix[0, 1]
            TP = current_confusion_matrix[1, 1]
            IoU = TP / np.maximum(1.0,(TP+FP+FN))
            avg_IoU.update(IoU)            

            qf1 = filename.split("_")[-3]
            qf2 = filename.split("_")[-1].split(".")[0]
            list_data.append(','.join((filename, qf1, qf2, str(mean_IoU), str(IoU))))
            

    f1_result = np.average(f1_array)
    prec_result = np.average(precission_array)
    recall_result = np.average(recall_array)

    confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size

    return print_loss, avg_mIoU.average(), avg_IoU.average(), IoU_array, pixel_acc, mean_acc, confusion_matrix, f1_result, prec_result, recall_result, list_data

def validate_combined(config, testloader, model, test_dataset):
    
    world_size = get_world_size()
    model.eval()
    
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    f1_array = np.array([])
    precission_array = np.array([])
    recall_array = np.array([])
    
    list_data = []
    ave_loss = AverageMeter()
    ave_loss_seg = AverageMeter()
    ave_loss_cls = AverageMeter()
    avg_mIoU = AverageMeter()
    avg_IoU = AverageMeter()
    precission = 0
    f1_image = 0
    valid_loss = 0.0
    valid_acc = 0.0
    correct,len_valset = 0, 0
    Path(mlflow.get_artifact_uri()[7:]+"/Masks/").mkdir(parents=True, exist_ok=True)
    Path(mlflow.get_artifact_uri()[7:]+"/Predictions/").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index, (image, mask, label, qtable) in enumerate(tqdm(testloader)):
            filename = get_next_filename(index, test_dataset)
            Image.fromarray((mask.cpu().numpy()[0] * 255).astype(np.uint8)).save(mlflow.get_artifact_uri()[7:]+"/Masks/"+filename.split(".")[-2]+".png")
            size = mask.size()
            image = image.cuda()
            mask = mask.long().cuda()
            label = label.long().cuda()
            
            losses, losses_seg, losses_cls, pred_seg, pred_cls = model(image, mask, label, qtable)            
            _, pred_class = torch.max(pred_cls, 1)
            pred_cls = torch.squeeze(pred_cls, 0)
            pred_cls = F.softmax(pred_cls, dim=0)[1]
            pred_cls = pred_cls.cpu().numpy()

            pred1_seg = torch.squeeze(pred_seg, 0)
            pred1_seg = F.softmax(pred1_seg, dim=0)[1]

            if index % 50 == 0:
                print(pred_class, int(label))
            if int(pred_class) == int(label):
                correct += 1
            len_valset = index

            pred_seg = F.upsample(input=pred_seg, size=(
                        size[-2], size[-1]), mode='bilinear')

            pred1_seg = pred1_seg.unsqueeze(0).unsqueeze(0)
            if pred1_seg.size()[-2] != size[-2] or pred1_seg.size()[-1] != size[-1]:
                pred1_seg = F.upsample(pred1_seg, (size[-2], size[-1]), mode='bilinear', align_corners=False)

            pred1_seg = pred1_seg.cpu().numpy().squeeze(axis=0).squeeze(axis=0)
            pred_mask = (pred1_seg >= 0.5)
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mlflow.get_artifact_uri()[7:]+"/Predictions/"+filename)

            loss = losses.mean()
            losses_seg = losses_seg.mean()
            losses_cls = losses_cls.mean()

            reduced_loss = reduce_tensor(loss)

            reduced_loss_seg = reduce_tensor(losses_seg)
            reduced_loss_cls = reduce_tensor(losses_cls)

            ave_loss.update(reduced_loss.item())
            ave_loss_seg.update(reduced_loss_seg.item())
            ave_loss_cls.update(reduced_loss_cls.item())

            if not(is_class_0(mask, size)):
                f1_image, precission, recall = get_f1_pixel_level(
                mask, 
                pred_seg, 
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
                
                f1_array = np.append(f1_array, f1_image)
                precission_array = np.append(precission_array, precission)
                recall_array = np.append(recall_array,recall)

            current_confusion_matrix = get_confusion_matrix(
                mask,
                pred_seg,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            confusion_matrix += current_confusion_matrix
            pos = current_confusion_matrix.sum(1)  # ground truth label count
            res = current_confusion_matrix.sum(0)  # prediction count
            tp = np.diag(current_confusion_matrix)  # Intersection part
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))  # Union part
            mean_IoU = IoU_array.mean()
            avg_mIoU.update(mean_IoU)
            TN = current_confusion_matrix[0, 0]
            FN = current_confusion_matrix[1, 0]
            FP = current_confusion_matrix[0, 1]
            TP = current_confusion_matrix[1, 1]
            IoU = TP / np.maximum(1.0,(TP+FP+FN))
            avg_IoU.update(IoU)
            list_data.append(','.join((filename, str(IoU), str(int(label)), str(pred_cls), str(int(pred_class)))))

    if correct != 0.0:
        valid_acc = 100 * correct / len_valset
        
    f1_result = np.average(f1_array)
    prec_result = np.average(precission_array)
    recall_result = np.average(recall_array)

    confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    print_loss = ave_loss.average()/world_size
    print_loss_seg = ave_loss_seg.average() / world_size
    print_loss_cls = ave_loss_cls.average() / world_size

    return valid_acc, print_loss, print_loss_seg, print_loss_cls, avg_IoU.average(), IoU_array, avg_mIoU.average(), pixel_acc, confusion_matrix, f1_result, prec_result, recall_result, list_data


def validate_cls(test_dataset, testloader, model):
    
    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    correct = 0
    len_valset = 0
    list_data = []
    with torch.no_grad():
        for batch_idx, (image, label, qtable) in enumerate(tqdm(testloader)):
            
            image, label = image.cuda(), label.long().cuda()
            model.eval()
            
            filename = get_next_filename(batch_idx, test_dataset)
            
            loss, output = model(image, label, qtable)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))
            _, pred1 = torch.max(output, 1)
            pred = torch.squeeze(output, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()

            if batch_idx % 50 == 0:
                print(pred, int(label))
            if int(pred1) == int(label):
                correct += 1
            len_valset = batch_idx

            list_data.append(','.join((filename, str(int(label)), str(pred))))
    if correct != 0.0:
        valid_acc = 100 * correct / len_valset
           
    return valid_loss, valid_acc, list_data

def get_next_filename(i, test_dataset):
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