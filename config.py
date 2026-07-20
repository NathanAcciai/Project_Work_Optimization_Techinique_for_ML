import utils
from utils import *


def get_model_dim(model):

    if hasattr(model, "embed_dim"):
        return model.embed_dim
    
    if hasattr(model, "hidden_dim"):
        return model.hidden_dim
    
    if hasattr(model, "fc"):
        return model.fc.in_features

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
            weight_decay=config["hyperparametres_general"].get("weight_decay"),
            n_heads=config.get("n_heads", 1),
            n_kv_heads=config.get("n_kv_heads", 1)
        )

       
        optimizer.adam_block_names.add("pos_embed")
        optimizer.adam_block_names.add("patch_embed")
        optimizer.wqk_names.add("qkv")
        optimizer.wv_names.add("qkv")
        optimizer.output_names.add("head")

        return optimizer
