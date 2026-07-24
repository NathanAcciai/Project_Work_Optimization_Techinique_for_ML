import utils
from utils import *


def get_model_dim(model):
    if hasattr(model, "embed_dim"):
        return model.embed_dim
    if hasattr(model, "hidden_dim"):
        return model.hidden_dim
    if hasattr(model, "fc"):
        return model.fc.in_features

def get_model_cfg(config, optimizer_name, model_name):
    opt_cfg = config[optimizer_name]
    if model_name=="ResNet-18":
        return opt_cfg["resnet"]
    elif model_name=="convnext_large":
        return opt_cfg["convnext"]
    else:
        return opt_cfg["vit"]

    




def build_optimizer(model, optimizer_name, model_name, config):
    cfg = get_model_cfg(config, optimizer_name, model_name)

    params = model.parameters()

    wd = cfg.get("weight_decay", 0)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params=params,
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            nesterov=cfg["nesterov"],
            weight_decay=cfg["weight_decay"]
    )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=params,
            lr=cfg["lr"],
            betas=tuple(cfg["betas"]),
            eps=cfg["eps"],
            weight_decay=cfg["weight_decay"]
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params=params,
            lr=cfg["lr"],
            betas=tuple(cfg["betas"]),
            eps=cfg["eps"],
            weight_decay=cfg["weight_decay"]
        )
    if optimizer_name == "adafactor":
        return Adafactor(
            params=params,
            lr=cfg["lr"],
            beta1=cfg["beta1"],
            decay_rate=cfg["decay_rate"],
            eps=tuple(cfg["eps"]),
            relative_step=cfg["relative_step"],
            scale_parameter=cfg["scale_parameter"],
            weight_decay=cfg["weight_decay"]
        )
    if optimizer_name == "lion":
        return Lion(
            params=params,
            lr=cfg["lr"],
            betas=tuple(cfg["betas"]),
            weight_decay=cfg["weight_decay"]
        )
    if optimizer_name == "adam-mini":
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

        optimizer = Adam_mini(
            model.named_parameters(),
            lr=cfg["lr"],
            dim=get_model_dim(model),
            betas=tuple(cfg["betas"]),
            eps=cfg["eps"],
            weight_decay=cfg["weight_decay"],
            n_heads=cfg.get("n_heads",1),
            n_kv_heads=cfg.get("n_kv_heads",1)
        )

        optimizer.adam_block_names.add("pos_embed")
        optimizer.adam_block_names.add("patch_embed")
        optimizer.wqk_names.add("qkv")
        optimizer.wv_names.add("qkv")
        optimizer.output_names.add("head")

        return optimizer


def build_scheduler(optimizer,  config):

    
    total_epochs = config["hyperparametres_general"]["epochs"]
    warmup_epochs = config["hyperparametres_general"]["warmup_epochs"]
    eta_min = config["hyperparametres_general"]["eta_min"]
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )


    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min
    )


    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            cosine_scheduler
        ],
        milestones=[warmup_epochs]
    )

    return scheduler