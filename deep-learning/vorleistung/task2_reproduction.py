import os
import random
import pickle
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from pathlib import Path
from torch.utils.data import Dataset
from tifffile import imread

from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime

# set root path and seed
PROJECT_ROOT = Path("/Users/luca/Projects/ms-data-science/deep-learning/vorleistung") #Path(os.getcwd()) #Path(r"C:\Users\tdoro\DLMS\mandatory_task")
DATASET_NAME = "EuroSAT_MS"
DATASET_ROOT = Path("/Users/luca/Projects/ms-data-science/deep-learning/vorleistung") / "data" #Path(os.getcwd()) / "data"

# use mat-nr as seed
RANDOM_SEED = 3778660
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

results_dir = PROJECT_ROOT / f"training_results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

#dataset class
class EuroSatMsDataset(Dataset):
    def __init__(self, dataset_path, split_file, transform=None):
        self.dataset_root_dir = dataset_path
        self.images = pd.read_csv(split_file)
        self.transform = transform

        # Map class name to idx
        self.classes = sorted(self.images.iloc[:, 1].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root_dir, self.images.iloc[idx, 0])

        label_name = self.images.iloc[idx, 1]
        label = torch.tensor(self.class_to_idx[label_name], dtype=torch.long)
        
        image = imread(img_path) # (H, W, C)
        image = image[:, :, [3, 2, 1]] # (H, W, 3)
        image = image.astype(np.float32) / 65535.0 # normalized to [0,1] # (H, W, 3)
        image = torch.from_numpy(image).permute(2, 0, 1) # (3, H, W)
        
        if self.transform:
            image = self.transform(image)

        return image, label

def predict_logits(model, dataloader, device):
    model.eval() # set model to inference mode

    with torch.no_grad(): # do not record computations for computing the gradient
    
        datasize = 0
        logits = []
  
        for ctr, data in enumerate(dataloader):
            inputs = data[0].to(device)
            outputs = model(inputs)
            cpuout = outputs.to('cpu')
            logits.append(cpuout)
          
            datasize += inputs.shape[0]
    
    return torch.cat(logits)

def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # end of for loop over augmentations/hyperparameters here!
    model.load_state_dict(weights_chosen)
    accuracy,_,tpr_per_class,logits = evaluate(model = model , dataloader  = dataloaders['test'], criterion = None, device = device)
    report = f'Final Result: val_acc={bestmeasure.item():.4f}, test_acc={accuracy.item():.4f}\n'

    class_names = dataloaders['train'].dataset.classes
    for i, cls_name in enumerate(class_names):
        report += f'{cls_name:<22}: {tpr_per_class[i]:.4f}\n'
    
    report += f'\nBest Hyperparameter: LR={best_hyperparameter}\n'
    report += f'Best Augmentation: {best_augmentation}\n'
    
    #torch.save(logits, results_dir / 'logits.pt')
    #pd.DataFrame({'image_path': dataloaders['test'].dataset.images.iloc[:, 0]}).to_csv(results_dir / 'logits.csv', index=False)


if __name__=='__main__':
  run()
    