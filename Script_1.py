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
            num_classes=10
        )
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name=="":
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
        config=yaml.safe_load(stream)
    os.makedirs("checkpoints", exist_ok=True)
    batch_sizes= config["hyperparametres_general"].get("batch_size")
    model_names= config["model_name"]
    optimizer_names= config["optimizer_name"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    datasets = ["cifar10", "cifar100"]
    os.makedirs("Results", exist_ok=True)
    
    for dataset_name in datasets:
        num_classes = 100 if dataset_name == "cifar100" else 10
        for bs in batch_sizes:
            train_dl, val_dl, test_dl = load_dataset(dataset_name = dataset_name, batch_size= bs)
            for model_name in model_names:
                model = Model_selection(num_classes,model_name).to(device)
                for opt_name in optimizer_names:
                    
                    optim= build_optimizer(model = model, optimizer_name = opt_name, config= config)
                    trainer = Trainer(config["hyperparametres_general"],model, optim)
                    run_name = f"{model_name}_{dataset_name}_{opt_name}_{bs}"
                    
                    print("\n" + "="*60)
                    print(f" Start Train: {run_name}")
                    print("="*60)

                    try:
                        
                        run = wandb.init(
                            project="Project_work_Optimization_Technique",
                            name=run_name,
                            group=f"{model_name}/{dataset_name}", # Crea una struttura a cartella su W&B
                            config={
                                "model": model_name,
                                "dataset": dataset_name,
                                "optimizer": opt_name,
                                "batch_size": bs,
                            },
                            reinit=True 
                        )
                        
                        trainer.training(train_dl, val_dl, run)
                        wandb.finish()
                        
                    except RuntimeError as e:
                        
                        if "out of memory" in str(e).lower():
                            print(f"\n[!!!] CUDA OUT OF MEMORY relevate: {run_name}")
                            print("[-] Step over this configuration, and sleep 10 sec")
                            
                            
                            if wandb.run is not None:
                                wandb.run.summary["status"] = "FAILED_OOM"
                                wandb.finish(exit_code=1) 
                            del model
                            del optimizer
                            del trainer
                            model = None
                            optimizer = None
                            trainer = None
                            
                            
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                            time.sleep(10)
                            print("[+] Clear Complete! \n")
                            continue 
                        
                        else:
                            
                            if wandb.run is not None:
                                wandb.finish(exit_code=1)
                            raise e

# %%
run_experiments()

# %%



