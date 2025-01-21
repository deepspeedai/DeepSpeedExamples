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
import torch
import os
from torchvision import transforms
from torchvision import datasets

def get_dataset(dataset_name, data_dir, split, rand_fraction=None,clean=False, transform=None, imsize=None, bucket='pytorch-data', **kwargs):

  if dataset_name in [ 'cifar10', 'cifar100']:
    dataset = globals()[f'get_{dataset_name}'](dataset_name, data_dir, split,  imsize=imsize, bucket=bucket, **kwargs)
  elif dataset_name in [ 'cifar10vit224', 'cifar100vit224','cifar10vit384', 'cifar100vit384',]:
    imsize = int(dataset_name.split('vit')[-1])
    dataset_name = dataset_name.split('vit')[0]
    #print ('here')
    dataset = globals()['get_cifar_vit'](dataset_name, data_dir, split, imsize=imsize, bucket=bucket, **kwargs)
  else:
    assert 'cifar' in dataset_name
  print (dataset_name)
  item = dataset.__getitem__(0)[0]

  dataset.nchannels = item.size(0)
  dataset.imsize = item.size(1)
  return dataset


def get_aug(split, imsize=None, aug='large'):
  if aug == 'large':
    imsize = imsize if imsize is not None else 224
    if split == 'train':
      return [transforms.Resize(round(imsize * 1.143)), transforms.RandomResizedCrop(imsize, scale=(0.2, 1.0)),transforms.RandomHorizontalFlip()]
      #return [transforms.Resize(round(imsize * 1.143)), transforms.CenterCrop(imsize)]
    else:
      return [transforms.Resize(round(imsize * 1.143)), transforms.CenterCrop(imsize)]
  else:
    imsize = imsize if imsize is not None else 32
    if split == 'train':
        train_transform = []
      #return [transforms.RandomCrop(imsize, padding=round(imsize / 8))]
        train_transform.append(transforms.RandomCrop(32, padding=4))
        train_transform.append(transforms.RandomHorizontalFlip())
        return train_transform
    else:
      return [transforms.Resize(imsize), transforms.CenterCrop(imsize)]


def get_transform(split, normalize=None, transform=None, imsize=None, aug='large'):
  if transform is None:
    if normalize is None:
        if aug == 'large':

          normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
          normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose(get_aug(split, imsize=imsize, aug=aug)
                                   + [transforms.ToTensor(), normalize])
  return transform


def get_cifar10(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  if imsize==224:
    transform = get_transform(split, transform=transform, imsize=imsize, aug='large')
  else:
    transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
  return datasets.CIFAR10(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)


def get_cifar100(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  if imsize==224:
    transform = get_transform(split, transform=transform, imsize=imsize, aug='large')
  else:
    transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
  return datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)

def get_cifar100N(dataset_name, data_dir, split, rand_fraction=None,transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
  if split=='train':
    return CIFAR100N(root=data_dir, train=(split=='train'), transform=transform, download=True, rand_fraction=rand_fraction)
  else:
    return datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)

def get_cifar_vit(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
    if imsize==224:

      if split=='train':
        transform_data = transforms.Compose([# transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
      else:
        transform_data = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
      if dataset_name =='cifar10':
          return datasets.CIFAR10(data_dir, train=(split=='train'), transform=transform_data, download=True, **kwargs)
      elif dataset_name =='cifar100':

          return datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform_data, download=True, **kwargs)
      else:
          assert dataset_name in ['cifar10', 'cifar100']
    else:

      if split=='train':
        transform_data = transforms.Compose([# transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
          transforms.Resize(imsize),
          transforms.RandomCrop(imsize, padding=(imsize//8)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
      else:
        transform_data = transforms.Compose([
          transforms.Resize(imsize),
          transforms.ToTensor(),
          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
      if dataset_name =='cifar10':
          return datasets.CIFAR10(data_dir, train=(split=='train'), transform=transform_data, download=True, **kwargs)
      elif dataset_name =='cifar100':

          return datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform_data, download=True, **kwargs)
      else:
          assert dataset_name in ['cifar10', 'cifar100']

def get_imagenet_vit(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):

    ngpus = torch.cuda.device_count()
    if split=='train':
      transform_data = transforms.Compose([# transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
      ])
    else:
      transform_data = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
      ])
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    if split == 'train':
      return datasets.ImageFolder(traindir, transform_data)
      #return torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
      return datasets.ImageFolder(valdir, transform_data)
      #Ereturn torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
