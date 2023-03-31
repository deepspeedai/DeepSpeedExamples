from __future__ import print_function
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from resnet import resnet
from tqdm import tqdm
import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean

# Training settings
parser = argparse.ArgumentParser(description='Training on Cifar10')

parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size',
                    type=int,
                    default=256,
                    metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--lr',
                    type=float,
                    default=0.1,
                    metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-decay',
                    type=float,
                    default=0.1,
                    help='learning rate ratio')
parser.add_argument('--lr-decay-epoch',
                    type=int,
                    nargs='+',
                    default=[4, 8],
                    help='Decrease learning rate at these epochs.')

parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--weight-decay',
                    default=5e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--batch-norm',
                    action='store_false',
                    help='do we need batch norm or not')
parser.add_argument('--residual',
                    action='store_false',
                    help='do we need residula connect or not')

parser.add_argument('--cuda',
                    action='store_false',
                    help='do we use gpu or not')
parser.add_argument('--saving-folder',
                    type=str,
                    default='checkpoints/',
                    help='choose saving name')
parser.add_argument('--compression',
                    action='store_true',
                    help='do we use compression or not')
parser.add_argument('--path-to-model',
                    type=str,
                    default=None)               

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

deepspeed.init_distributed()

# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# get dataset
train_loader, test_loader = getData(name='cifar10',
                                    train_bs=args.batch_size,
                                    test_bs=args.test_batch_size)

# get model and optimizer
model = resnet(num_classes=10,
               depth=20,
               residual_not=args.residual,
               batch_norm_not=args.batch_norm)
if args.cuda:
    model = model.cuda()

if args.compression:
    assert args.path_to_model is not None
    model.load_state_dict(torch.load(args.path_to_model))
    model = init_compression(model, args.deepspeed_config)
    
criterion = nn.CrossEntropyLoss()
model_engine, optimizer, _, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=model.parameters())


if not os.path.isdir(args.saving_folder):
    os.makedirs(args.saving_folder)

for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    with tqdm(total=len(train_loader.dataset)) as progressbar:
        for batch_idx, (data, target) in enumerate(train_loader):

            model_engine.train()
            if args.cuda:
                data, target = data.cuda().half(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            model_engine.backward(loss)
            train_loss += loss.item() * target.size()[0]
            total_num += target.size()[0]
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            model_engine.step()

            progressbar.set_postfix(loss=train_loss / total_num,
                                    acc=100. * correct / total_num)

            progressbar.update(target.size(0))

    acc = test(epoch, model, test_loader, fp16=True)
if args.compression:
    model = redundancy_clean(model, args.deepspeed_config)
    print ('after_clean')
    acc = test(epoch, model, test_loader, fp16=True)
    torch.save(model.state_dict(), args.saving_folder + 'clean_net.pkl')
else:
    torch.save(model.state_dict(), args.saving_folder + 'net.pkl')