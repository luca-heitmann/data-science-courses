import os
import random
import pickle
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet18
from pathlib import Path
from torch.utils.data import Dataset
from tifffile import imread
from config import PROJECT_ROOT, DATASET_PATH, SEED, REPRODUCTION_MODEL_PATH

# set root path and seed
DATASET_NAME = "EuroSAT_MS"

# use mat-nr as seed
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

reproduction_dir = Path(PROJECT_ROOT) / "reproduction"

os.makedirs(reproduction_dir, exist_ok=True)

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

#model for task 2
class EuroSatResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        #load pretrained model
        self.backbone = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # Adapt avgpool to handle small input sizes (like 64x64 -> 2x2 feature map)
        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        
        #replace last layer according to our classification classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x): # x shape: (batch_size, 3, H, W) bzw. (batch_size, C, H, W) bzw. (32, 3, 64, 64)
        return self.backbone(x)

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

import argparse

def run():
    parser = argparse.ArgumentParser(description='Task 2 Reproduction')
    parser.add_argument('--generate-logits', action='store_true', help='Generate and save logits instead of checking them')
    args = parser.parse_args()
    
    generate_logits = args.generate_logits

    batchsize = 32

    with open(REPRODUCTION_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = EuroSatMsDataset(DATASET_PATH, Path(PROJECT_ROOT) / "data_splits" / DATASET_NAME / "test.csv")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    logits = predict_logits(model, test_dataloader, device)
    
    if generate_logits:
        torch.save(logits, reproduction_dir / 'logits.pt')
        pd.DataFrame({'image_path': test_dataset.images.iloc[:, 0]}).to_csv(reproduction_dir / 'logits.csv', index=False)

        print("Logits saved")
    else:
        prev_logits = torch.load(reproduction_dir / 'logits.pt')
        assert torch.allclose(prev_logits, logits)

        print("Logits match")

if __name__=='__main__':
  run()
    