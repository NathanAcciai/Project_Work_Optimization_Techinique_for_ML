# %%

import utils

import Trainer
import config
from utils import *

from config import *
from Trainer import *

# %%
def Model_selection(num_classes, model_name= "ResNet-18" ):
    if model_name=="ResNet-18":
        model = resnet18(
            weights=None,
            num_classes=num_classes
        )

    elif model_name=="ViT-Tiny":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=num_classes
        )
    else:
        #da vedere
        model= None
    return model




# %%
#model= Model_selection(10, "ResNet-18")




# %%
def run_experiments():
    
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)
    
    batch_sizes = config["hyperparametres_general"].get("batch_size")
    model_names = config["model_name"]
    optimizer_names = config["optimizer_name"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    datasets = ["cifar10", "cifar100"]
    
    for dataset_name in datasets:
        num_classes = 100 if dataset_name == "cifar100" else 10
        dataset_config = get_config_for_dataset(config, dataset_name)
        for bs in batch_sizes:
            for model_name in model_names:
                for opt_name in optimizer_names:
                    
                    # 1. Ricreiamo i DataLoader freschi per ogni esperimento (evita deadlock sui worker)
                    
                    train_dl, val_dl, test_dl = load_dataset(dataset_name=dataset_name, batch_size=bs)
                    
                    # 2. Ricreiamo il MODELLO da zero per ogni ottimizzatore!
                    model = Model_selection(model_name=model_name, num_classes=dataset_config["num_classes"]).to(device)
                    optimizer = build_optimizer(model, opt_name, model_name, dataset_config)
                    scheduler = build_scheduler(optimizer, opt_name, model_name, dataset_config)

                    
                    
                    run_name = f"{model_name}_{dataset_name}_{opt_name}_{bs}"
                    path_checkpoint = f"checkpoints/{run_name}"
                    done_flag = os.path.join(path_checkpoint, "DONE")
                    os.makedirs(path_checkpoint, exist_ok=True)
                    if os.path.exists(done_flag):
                        print(f"[SKIP] {run_name} alredy completed.")
                        continue
                    
                    trainer = Trainer(config=dataset_config["hyperparametres_general"],
                                          model=model,
                                          optimizer=optimizer,        
                                          scheduler=scheduler,        
                                          checkpoint_path=path_checkpoint,        
                                          patience=dataset_config["hyperparametres_general"]["patience_tiny"] if model_name == "ViT-Tiny"
                                                    else dataset_config["hyperparametres_general"]["patience_resnet"],        
                                          model_name=model_name    
                                          )
                    
                    print("\n" + "="*60)
                    print(f" Start Train: {run_name}")
                    print("="*60)

                    try:
                        run = wandb.init(
                            project="Project_work_Optimization_Technique",
                            name=run_name,
                            group=f"{model_name}-{dataset_name}-bs{bs}",
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
                            if wandb.run is not None:
                                wandb.run.summary["status"] = "FAILED_OOM"
                                wandb.finish(exit_code=1)
                        else:
                            if wandb.run is not None:
                                wandb.finish(exit_code=1)
                            raise e

                    finally:
                        # 3. PULIZIA COMPLETA: viene eseguita SEMPRE a fine run
                        del model, optim, trainer, train_dl, val_dl, test_dl
                        if 'scheduler' in locals() and scheduler is not None:
                            del scheduler
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                        time.sleep(3)

# %%
run_experiments()

# %%



