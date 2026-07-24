# %%

import utils

import Trainer
import config
from utils import *

from config import *
from Trainer import *

# %%
def Model_selection(num_classes, config,model_name= "ResNet-18" ):
    if model_name=="ResNet-18":
        #dattata a cifar
        model = resnet18(
            weights=None,
            num_classes=num_classes
        )

        model.conv1 = nn.Conv2d(
            3,64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        model.maxpool = nn.Identity()

    elif model_name=="ViT-Tiny":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            img_size=32,
            patch_size=4,
            num_classes=num_classes,
            drop_path_rate=config["vit_specific"]["drop_path_rate"]
        ) 
    elif model_name=="ViT-Large":
        model = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            patch_size= 8,
            num_classes=200, 
            img_size=64, 
            drop_path_rate=0.2
            )
    elif model_name=="convnext_large":
        model =timm.create_model(
            'convnext_large', 
            pretrained=False, 
            num_classes=200
            )

        return model

    else:
        #da vedere
        model= None
    return model




# %%
#model= Model_selection(10, "ResNet-18")




# %%
def run_experiments(single_experiments=True):
    if single_experiments:
        with open("config_2.yaml") as stream:
                config = yaml.safe_load(stream)
    else:
        with open("config.yaml") as stream:
            config = yaml.safe_load(stream)
    
    batch_sizes = config["hyperparametres_general"].get("batch_size")
    model_names = config["model_name"]
    optimizer_names = config["optimizer_name"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    datasets = ["cifar10", "cifar100","tiny_imagenet"]
    
    for dataset_name in datasets:
        if "cifar10" in dataset_name:
            continue
        for bs in batch_sizes:
            for model_name in model_names:
                for opt_name in optimizer_names:
                    train_dl, val_dl, test_dl = load_dataset(dataset_name=dataset_name, batch_size=bs)
                    
                    
                    model = Model_selection(model_name=model_name, config=config,num_classes=config["datasets"][dataset_name]["num_classes"]).to(device)
                    optimizer = build_optimizer(model,
                                                opt_name,
                                                model_name,
                                                config
                                            )
                    scheduler = build_scheduler(optimizer,
                                                config
                                            )

                    
                    if single_experiments:
                        run_name = f"{model_name}_{dataset_name}_{opt_name}_{bs}_test"
                    else:
                        run_name = f"{model_name}_{dataset_name}_{opt_name}_{bs}"
                    path_checkpoint = f"checkpoints/{run_name}"
                    done_flag = os.path.join(path_checkpoint, "DONE")
                    os.makedirs(path_checkpoint, exist_ok=True)
                    if os.path.exists(done_flag):
                        print(f"[SKIP] {run_name} alredy completed.")
                        continue
                    config_patience= config["datasets"][dataset_name]
                    if model_name=="ResNet-18":
                        patience= config_patience["patience_resnet"]
                    elif model_name== "ViT-Tiny":
                        patience= config_patience["patience_tiny"]
                    elif model_name=="convnext_large":
                        patience= config_patience["patience_convnext"]
                    else:
                        patience= config_patience["patience_vit"]
                    trainer = Trainer(config=config["hyperparametres_general"],
                                          model=model,
                                          optimizer=optimizer,        
                                          scheduler=scheduler,        
                                          checkpoint_path=path_checkpoint,        
                                          patience=patience,
                                          model_name=model_name    
                                          )
                    
                    print("\n" + "="*60)
                    print(f" Start Train: {run_name}")
                    print("="*60)

                    try:
                        run = wandb.init(
                            project="Project_work_Optimization_Technique",
                            name=run_name,
                            group=f"{model_name}-{dataset_name}-bs{bs}_test",
                            config={
                                "model": model_name,
                                "dataset": dataset_name,
                                "optimizer": opt_name,
                                "batch_size": bs,
                            },
                            reinit=True 
                        )
                        
                        trainer.training(train_dl, val_dl, run)
                        trainer.test(test_dl, run)
                        wandb.finish()
                        with open(done_flag, "w") as f:
                            f.write("ok")

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"\n[!!!] CUDA OUT OF MEMORY rilevato: {run_name}")
                            with open(done_flag, "w") as f:
                                f.write("ok")
                            if wandb.run is not None:
                                wandb.run.summary["status"] = "FAILED_OOM"
                                wandb.finish(exit_code=1)
                        else:
                            if wandb.run is not None:
                                wandb.finish(exit_code=1)
                            raise e
                        del model, optimizer, trainer, train_dl, val_dl, test_dl
                        if 'scheduler' in locals() and scheduler is not None:
                            del scheduler
                        
                        gc.collect()

                    finally:
                        # 3. PULIZIA COMPLETA: viene eseguita SEMPRE a fine run
                        
                        torch.cuda.empty_cache()
                        time.sleep(3)

# %%
run_experiments(True)

# %%



