import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
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
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.models import resnet18
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
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
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
    else:
        raise ValueError("Scegli tra 'CIFAR10' o 'CIFAR100'")
    
    train_size = int(0.8 * len(train))
    val_size = len(train) - train_size
    train_dataset, val_dataset = random_split(train, [train_size, val_size])
    train_dl= DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_dl= DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
    test_dl= DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)

    return train_dl, val_dl, test_dl
        

