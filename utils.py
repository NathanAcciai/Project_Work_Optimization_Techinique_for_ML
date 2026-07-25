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
from PIL import Image
import gc
from pathlib import Path
import zipfile
import urllib.request
import random

import torch.nn as nn
import torch.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.models import resnet18, vit_b_16
from transformers.optimization import Adafactor
from lion_pytorch import Lion
from adam_mini import Adam_mini
from transformers import CLIPModel


# ---------------------------------------------------------------------------
# Utility generiche
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Se si usano più GPU

    # Garantisce determinismo su CUDA / cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# ---------------------------------------------------------------------------
# CIFAR-10 / CIFAR-100
# ---------------------------------------------------------------------------

def load_dataset_cifar(dataset_name="cifar10", batch_size=16):
    # standard trasformation for this type of dataset

    if "cifar" in dataset_name:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        crop = 32

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset_name.upper() == "CIFAR10":
        train = CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=train_transform
        )
        test = CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=test_transform
        )

    elif dataset_name.upper() == "CIFAR100":
        train = CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=train_transform
        )

        test = CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=test_transform
        )

    else:
        raise ValueError("Scegli tra 'CIFAR10' o 'CIFAR100' o 'Imagenet200'")

    train_size = int(0.8 * len(train))
    val_size = len(train) - train_size

    train_dataset, val_dataset = random_split(train, [train_size, val_size])
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Tiny-ImageNet
# ---------------------------------------------------------------------------

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_tiny_imagenet(data_dir: str) -> str:

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / "tiny-imagenet-200"

    if dataset_dir.exists():
        print(f"Dataset già presente in {dataset_dir}")
        return str(dataset_dir)

    zip_path = data_dir / "tiny-imagenet-200.zip"
    if not zip_path.exists():
        print(f"Download da {TINY_IMAGENET_URL} ...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download completato.")

    print("Estrazione in corso...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print(f"Estratto in {dataset_dir}")

    return str(dataset_dir)


class TinyImageNetTrain(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root) / "train"
        self.transform = transform

        classes = sorted(os.listdir(self.root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.samples = []
        for cls in classes:
            img_dir = self.root / cls / "images"
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((str(img_dir / fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TinyImageNetVal(Dataset):
    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.root = Path(root) / "val"
        self.img_dir = self.root / "images"
        self.transform = transform
        self.class_to_idx = class_to_idx

        ann_path = self.root / "val_annotations.txt"
        self.samples = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, wnid = parts[0], parts[1]
                self.samples.append((str(self.img_dir / fname), self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TinyImageNetTest(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.samples = [base_dataset.samples[i] for i in indices]
        self.transform = transform

        self.class_to_idx = base_dataset.class_to_idx
        self.idx_to_class = base_dataset.idx_to_class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def build_transforms(img_size: int, train: bool, pretrained_norm: bool):
    mean = [0.485, 0.456, 0.406] if pretrained_norm else [0.4802, 0.4481, 0.3975]
    std = [0.229, 0.224, 0.225] if pretrained_norm else [0.2770, 0.2691, 0.2821]

    ops = []
    if train:
        ops += [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=img_size // 8),
        ]
    else:
        ops += [transforms.Resize((img_size, img_size))]

    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(ops)


def get_dataloaders_tiny(
    data_dir: str = "./data",
    img_size: int = 64,
    batch_size: int = 128,
    pretrained_norm: bool = True,
    num_workers: int = 4,
):
    dataset_root = download_tiny_imagenet(data_dir)

    train_tf = build_transforms(
        img_size,
        train=True,
        pretrained_norm=pretrained_norm
    )

    val_tf = build_transforms(
        img_size,
        train=False,
        pretrained_norm=pretrained_norm
    )

    full_train = TinyImageNetTrain(
        dataset_root,
        transform=train_tf
    )

    full_train_eval = TinyImageNetTrain(
        dataset_root,
        transform=val_tf
    )

    # Split
    test_size = int(0.1 * len(full_train))
    train_size = len(full_train) - test_size

    generator = torch.Generator().manual_seed(42)

    train_set, test_set = torch.utils.data.random_split(
        full_train,
        [train_size, test_size],
        generator=generator
    )

    test_set = torch.utils.data.Subset(
        full_train_eval,
        test_set.indices
    )

    # Validation ufficiale
    val_set = TinyImageNetVal(
        dataset_root,
        class_to_idx=full_train.class_to_idx,
        transform=val_tf
    )

    # Loader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"Train: {len(train_set)} | "
        f"Val: {len(val_set)} | "
        f"Test: {len(test_set)} | "
        f"Classes: {len(full_train.class_to_idx)}"
    )

    return train_loader, val_loader, test_loader