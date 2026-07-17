
import utils
from utils import *
class Earlystopping():
    def __init__(self, patience, min_delta = 0.005, ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None


    def __call__(self, current_score, model):

        if self.best_score is None:
            self.best_score = current_score
            return True 
        improved = current_score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.counter = 0 
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        


class Trainer():
    def __init__(self, config, model , optimizer):
        super(Trainer,self).__init__()
        self.optimizer = optimizer
        self.scheduler= None
        self.epochs = config["epochs"]
        self.model= model
        self.patience= config["patience"]
        self.criterion = nn.CrossEntropyLoss()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.best_model= None
        self.val_accuracy = -np.inf 
        self.best_accuracy= 0   
        self.early_stopping = Earlystopping(self.patience)
    
    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0
        total =0 
        correct = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        
        start_time= time.time()
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        end_time= time.time()-start_time
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_memory_mb = peak_memory_bytes / (1024 ** 2)
        
        return epoch_loss, epoch_acc, peak_memory_mb, end_time
    
    
    @torch.no_grad()
    def evaluate(self,dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_memory_mb = peak_memory_bytes / (1024 ** 2)

        val_loss = running_loss / total
        val_acc = 100 * correct / total
        return val_loss, val_acc, peak_memory_mb
    


    def training(self, dataloader_train, dataloader_val,wandb_run):
        self.best_accuracy=0
        start_time= time.time()
        for epoch in tqdm(range(self.epochs), desc= "Train Epochs"):

            train_loss, train_acc, train_mem, train_time= self.train_one_epoch(dataloader_train)
            val_loss, val_acc, val_mem = self.evaluate(dataloader = dataloader_val)

            time_data = format_elapsed_time(train_time)
            
            
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "system/train_peak_gpu_memory_mb": train_mem,
                "system/val_peak_gpu_memory_mb": val_mem,
                "time_total/seconds": train_time,
                "time_total/formatted": time_data["string"],
                "time_total/days": time_data["days"],
                "time_total/hours": time_data["hours"],
                "time_total/minutes": time_data["minutes"],
                "time_total/seconds_remain": time_data["seconds"]
            })
            
            # Stampa di controllo
            print(f"Epoch {epoch:02d}/{self.epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}% | "
                  f"Mem: {train_mem:.1f} MB | Time: {train_time:.2f}s")
            
            
            if  self.early_stopping and epoch > 20:
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'best_accuracy': self.best_accuracy,
                }
                torch.save(checkpoint_data, "checkpoints/best_model.pt")
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                
            
            if self.scheduler is not None:
                self.scheduler.step()
                
        self.model.load_state_dict(self.best_model_wts)
        

        wandb_run.run.summary["best_val_accuracy"] = self.best_accuracy
        wandb_run.run.summary["total_training_time"] = format_elapsed_time(time.time() - start_time)["string"]
        
        
    
    def Test(self, dataloader,wandb_run):
        start_test= time.time()
        _,test_acc, test_mem = self.evaluate(dataloader = dataloader)
        end_test= time.time()- start_test
        time_data= format_elapsed_time(end_test)
        wandb_run.log({
            "test/accuracy_test": test_acc,
            "test/peak_gpu_memory_mb": test_mem,
            "test/time_total/formatted": time_data["string"],
            "test/time_total/days": time_data["days"],
            "test/time_total/hours": time_data["hours"],
            "test/time_total/minutes": time_data["minutes"],
            "test/time_total/seconds_remain": time_data["seconds"]
        })

