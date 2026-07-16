# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorboard
import wandb
import transformers
import lion_pytorch
import adam_mini
import tqdm
import yaml
import timm 
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.models import resnet18
from transformers.optimization import Adafactor
from lion_pytorch import Lion
from adam_mini import Adam_mini
from transformers import CLIPModel




# %%
def load_dataset(name_dataset="cifar10"):
    if name_dataset=="cifar10":
        train = CIFAR10(
            root="./data",
            train=True,
            download=True
        )

        test = CIFAR10(
            root="./data",
            train=False,
            download=True
        )
    else:
        train = CIFAR100(
            root="./data",
            train=True,
            download=True
        )

        test = CIFAR100(
            root="./data",
            train=False,
            download=True
        )
    return train, test
        



# %%
cifar100= load_dataset("S")

# %%
def Model_definition(model_name= "resnet"):
    if model_name=="resnet":
            model = resnet18(
            weights=None,
            num_classes=10
        )
    elif model_name=="":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=10
        )
    else:
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        model= model.vision_model
        model.classifier = nn.Linear(
            768,
            100
        )
    return model


# %%



