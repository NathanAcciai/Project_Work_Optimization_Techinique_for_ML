import utils
from utils import *


def get_model_dim(model):
    if hasattr(model, "embed_dim"):
        return model.embed_dim
    if hasattr(model, "hidden_dim"):
        return model.hidden_dim
    if hasattr(model, "fc"):
        return model.fc.in_features


def get_weight_decay(model_name, config):
    """Seleziona il weight_decay corretto in base al modello."""
    gen_cfg = config["hyperparametres_general"]
    if model_name == "ViT-Tiny":
        return gen_cfg.get("weight_decay_vit")
    return gen_cfg.get("weight_decay_resnet")


def build_optimizer(model, optimizer_name, model_name, config):
    cfg = config[optimizer_name]
    params = model.parameters()
    wd = get_weight_decay(model_name, config)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params=params,
            lr=cfg.get("lr"),
            momentum=cfg.get("momentum"),
            nesterov=cfg.get("nesterov"),
            weight_decay=wd
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=params,
            lr=cfg.get("lr"),
            betas=tuple(cfg.get("betas")),
            eps=cfg.get("eps"),
            weight_decay=0  # adam tenuto a 0 per design, come da tuo yaml originale
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params=params,
            lr=cfg.get("lr"),
            betas=tuple(cfg.get("betas")),
            eps=cfg.get("eps"),
            weight_decay=wd
        )
    if optimizer_name == "adafactor":
        return Adafactor(
            params=params,
            lr=cfg.get("lr"),
            beta1=cfg.get("beta1"),
            decay_rate=cfg.get("decay_rate"),
            eps=tuple(cfg.get("eps")),
            relative_step=cfg.get("relative_step"),
            scale_parameter=cfg.get("scale_parameter"),
            weight_decay=wd
        )
    if optimizer_name == "lion":
        return Lion(
            params=params,
            lr=cfg.get("lr"),
            betas=tuple(cfg.get("betas")),
            weight_decay=wd
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
            lr=cfg.get("lr"),
            dim=get_model_dim(model),
            betas=tuple(cfg.get("betas")),
            eps=cfg.get("eps"),
            weight_decay=wd,
            n_heads=cfg.get("n_heads", 1),     
            n_kv_heads=cfg.get("n_kv_heads", 1) 
        )

        optimizer.adam_block_names.add("pos_embed")
        optimizer.adam_block_names.add("patch_embed")
        optimizer.wqk_names.add("qkv")
        optimizer.wv_names.add("qkv")
        optimizer.output_names.add("head")

        return optimizer


def build_scheduler(optimizer, optimizer_name, model_name, config):
    total_epochs = config["hyperparametres_general"]["epochs"]

    if model_name == "ViT-Tiny":
        vit_cfg = config["vit_specific"]
        warmup_epochs = vit_cfg.get("warmup_epochs", 0)

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    
    if optimizer_name == "sgd":
        cfg = config["sgd"]
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("T_max"), eta_min=cfg.get("eta_min")
        )

    return None