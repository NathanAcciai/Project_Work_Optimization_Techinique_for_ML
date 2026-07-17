import utils
from utils import *


def build_optimizer(model, optimizer_name,config):
    cfg= config[optimizer_name]
    params= model.parameters()

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params= params,
            lr= cfg.get("lr"),
            momentum= cfg.get("momentum"),
            nesterov= cfg.get("nesterov"),
            weight_decay = config["hyperparametres_general"].get("weight_decay")
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            params= params,
            lr= cfg.get("lr"),
            betas= tuple(cfg.get("betas")),
            eps= cfg.get("eps"),
            weight_decay=0
        )
    if optimizer_name== "adamw":
        return torch.optim.AdamW(
            params= params,
            lr = cfg.get("lr"),
            betas= tuple(cfg.get("betas")),
            eps= cfg.get("eps"),
            weight_decay=config["hyperparametres_general"].get("weight_decay")
        )
    if optimizer_name=="adafactor":
        return Adafactor(
            params= params,
            lr= cfg.get("lr"),
            beta1= cfg.get("beta1"),
            decay_rate= cfg.get("decay_rate"),
            eps= tuple(cfg.get("eps")),
            relative_step= cfg.get("relative_step"),
            scale_parameter= cfg.get("scale_parameter"),
            weight_decay=config["hyperparametres_general"].get("weight_decay")
            
        )
    if optimizer_name =="lion":
        return Lion(
            params=params,
            lr=cfg.get("lr"),
            betas= tuple(cfg.get("betas")),
            weight_decay=config["hyperparametres_general"].get("weight_decay")
        )
    if optimizer_name=="adam-mini":
        return Adam_mini(
            params,
            lr=cfg.get("lr"),
            betas=tuple(cfg.get("betas")),
            eps= cfg.get("eps"),
            weight_decay=config["hyperparametres_general"].get("weight_decay")
        )   
