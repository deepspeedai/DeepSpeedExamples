# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import torch
from torch import Tensor
import torch.nn as nn
import torch.backends.cudnn as cudnn
import subprocess
import os
import time
import shutil
from datetime import datetime
import torch.optim as optim
from torch.optim import lr_scheduler
import sys
sys.path.append("..")
import models
import numpy as np
import torch.nn as nn
import math

def run_cmd(cmd_str, prev_sp=None):
  """
  This function runs the linux command cmr_str as a subprocess after waiting
  for prev_sp subprocess to finish
  """
  if prev_sp is not None:
    prev_sp.wait()
  return subprocess.Popen(cmd_str, shell=True)#, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))


def get_model(model_name, nchannels=3, imsize=32, nclasses=10, half=False):

  ngpus = torch.cuda.device_count()

  print("=> creating model '{}'".format(model_name))
  if imsize < 128 and model_name in models.__dict__:
    model = models.__dict__[model_name](num_classes=nclasses, nchannels=nchannels)
  else:
    model = models.__dict__[model_name](num_classes=nclasses, nchannels=nchannels)
  model = nn.DataParallel(model).cuda()
  cudnn.benchmark = True
  if half:
    print('Using half precision except in Batch Normalization!')
    model = model.half()
    for module in model.modules():
      if (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)):
        module.float()
  return model

def get_optimizer(optimizer_name, parameters, lr, momentum=0, weight_decay=0):
  if optimizer_name == 'sgd':
    return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay)
  elif optimizer_name == 'nesterov_sgd':
    return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
  elif optimizer_name == 'rmsprop':
    return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
  elif optimizer_name == 'adagrad':
    return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif optimizer_name == 'adam':
    return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

def get_scheduler(scheduler_name, optimizer, num_epochs, **kwargs):
  if scheduler_name == 'constant':
    return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)
  elif scheduler_name == 'step2':
    return lr_scheduler.StepLR(optimizer, round(num_epochs / 2), gamma=0.1, **kwargs)
  elif scheduler_name == 'step3':
    return lr_scheduler.StepLR(optimizer, round(num_epochs / 3), gamma=0.1, **kwargs)
  elif scheduler_name == 'exponential':
    return lr_scheduler.ExponentialLR(optimizer, (1e-3) ** (1 / num_epochs), **kwargs)
  elif scheduler_name == 'cosine':
    return lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, **kwargs)
  elif scheduler_name == 'step-more':
    return lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2, **kwargs)

def run_cmd(cmd_str, prev_sp=None):
  """
  This function runs the linux command cmr_str as a subprocess after waiting
  for prev_sp subprocess to finish
  """
  if prev_sp is not None:
    prev_sp.wait()
  return subprocess.Popen(cmd_str, shell=True)#, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))


class LossTracker(object):
  def __init__(self, num, prefix='', print_freq=1):
    self.print_freq=print_freq
    self.batch_time = AverageMeter('Time', ':6.3f')
    self.losses = AverageMeter('Loss', ':.4e')
    self.top1 = AverageMeter('Acc@1', ':6.2f')
    self.top5 = AverageMeter('Acc@5', ':6.2f')
    self.progress = ProgressMeter( num, [self.batch_time, self.losses, self.top1, self.top5], prefix=prefix)
    self.end = time.time()

  def update(self, loss, output, target):
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    self.losses.update(loss.item(), output.size(0))
    self.top1.update(acc1[0], output.size(0))
    self.top5.update(acc5[0], output.size(0))

  def display(self, step):
    self.batch_time.update(time.time() - self.end)
    self.end = time.time()
    if step % self.print_freq == 0:
      self.progress.display(step)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

