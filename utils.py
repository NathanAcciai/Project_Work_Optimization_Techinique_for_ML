import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import tensorboard
import wandb
import transformers
import lion_pytorch
import adam_mini
import copy
from tqdm.notebook import tqdm
import torch.distributed as dist
import yaml
import timm 
import time

import gc

import torch.nn as nn
import torch.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.models import resnet18,vit_b_16
from transformers.optimization import Adafactor
from lion_pytorch import Lion
from adam_mini import Adam_mini
from transformers import CLIPModel

def format_elapsed_time(seconds):
    
    days = int(seconds // (24 * 3600))
    seconds %= (24 * 3600)
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    time_str = " ".join(parts)
    
    return {
        "string": time_str,
        "days": days,
        "hours": hours,
        "minutes": minutes,
        "seconds": secs
    }


def load_dataset(dataset_name="cifar10",batch_size= 16):
    #standard trasformation for this type of dataset
    

    if "cifar" in dataset_name:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        crop= 32
    else:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        crop= 64
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64,scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2,magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    if dataset_name.upper() == "CIFAR10":
        train = CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform= train_transform
        )
        test = CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform= test_transform
        )
        val= None
    elif dataset_name.upper() == "CIFAR100":
        train = CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform= train_transform
        )

        test = CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform= test_transform
        )
        val=None
    elif dataset_name.upper() == "TINY_IMAGENET":
        path= "./data/tiny-imagenet-200"
        train = datasets.ImageFolder(
            path+"/train",
            train_transform
        )

        val = datasets.ImageFolder(
            path+"/val",
            val_transform
        )   
        test = datasets.ImageFolder(
                    path+"/test",
                    val_transform
                )    
    else:
        raise ValueError("Scegli tra 'CIFAR10' o 'CIFAR100' o 'Imagenet200'")
    
    train_size = int(0.8 * len(train))
    val_size = len(train) - train_size
    if val is None:
        train_dataset, val_dataset = random_split(train, [train_size, val_size])
    else:
        train_dataset= train
        val_dataset= val
    train_dl= DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_dl= DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
    test_dl= DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)

    return train_dl, val_dl, test_dl


def get_config_for_dataset(config, dataset_name):
    cfg = copy.deepcopy(config)
    dataset_cfg = config["datasets"][dataset_name]

    cfg["hyperparametres_general"]["patience_resnet"] = dataset_cfg["patience_resnet"]
    cfg["hyperparametres_general"]["patience_tiny"] = dataset_cfg["patience_tiny"]
    cfg["hyperparametres_general"]["weight_decay_resnet"] = dataset_cfg["weight_decay_resnet"]
    cfg["hyperparametres_general"]["weight_decay_vit"] = dataset_cfg["weight_decay_vit"]

    # merge vit_specific (override solo le chiavi presenti, mantiene le altre)
    cfg["vit_specific"].update(dataset_cfg.get("vit_specific", {}))

    cfg["num_classes"] = dataset_cfg["num_classes"]
    return cfg
        

