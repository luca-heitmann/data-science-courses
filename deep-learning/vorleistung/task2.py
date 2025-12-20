import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from pathlib import Path
from torch.utils.data import Dataset
from tifffile import imread

# set root path and seed
PROJECT_ROOT = Path(os.getcwd()) #Path(r"C:\Users\tdoro\DLMS\mandatory_task")
DATASET_NAME = "EuroSAT_MS"
DATASET_ROOT = Path(os.getcwd()) / "data"

# use mat-nr as seed
RANDOM_SEED = 3778660
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#dataset class
class EuroSatMsDataset(Dataset):
    def __init__(self, dataset_path, split_file):
        self.dataset_root_dir = dataset_path
        self.img_labels = pd.read_csv(split_file)

        # Map class name to idx
        self.classes = sorted(self.img_labels.iloc[:, 1].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root_dir, self.img_labels.iloc[idx, 0])

        label_name = self.img_labels.iloc[idx, 1]
        label = torch.tensor(self.class_to_idx[label_name], dtype=torch.long)
        
        image = imread(img_path) # (H, W, C)
        image = image[:, :, [3, 2, 1]] # (H, W, 3)
        image = image.astype(np.float32) / 65535.0 # normalized to [0,1] # (H, W, 3)
        image = torch.from_numpy(image).permute(2, 0, 1) # (3, H, W)
        
        return image, label

#model for task 2
class EuroSatResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        #load base model
        self.backbone = resnet18(pretrained=pretrained)
        
        #replace last layer according to our classification classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x): # x shape: (batch_size, 3, H, W) bzw. (batch_size, C, H, W) bzw. (32, 3, 64, 64)
        return self.backbone(x)
    
def train_epoch(model,  trainloader,  criterion, device, optimizer ):

    model.train() # IMPORTANT!!!
 
    losses = []
    for batch_idx, data in enumerate(trainloader):

        inputs=data[0].to(device)
        labels=data[1].to(device)
       

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad() #reset accumulated gradients  
        loss.backward() #compute new gradients
        optimizer.step() # apply new gradients to change model parameters

        losses.append(loss.item())

    return losses

def evaluate(model, dataloader, criterion, device):

    model.eval() # set model to inference mode


    with torch.no_grad(): # do not record computations for computing the gradient
    
      datasize = 0
      accuracy = 0
      avgloss = 0
      for ctr, data in enumerate(dataloader):

          #print ('epoch at',len(dataloader.dataset), ctr)
          
          inputs = data[0].to(device)        
          outputs = model(inputs)

          labels = data[1]

          # computing some loss
          cpuout= outputs.to('cpu')
          if criterion is not None:
            curloss = criterion(cpuout, labels)
            avgloss = ( avgloss*datasize + curloss ) / ( datasize + inputs.shape[0])

          # for computing the accuracy
          labels = labels.float()
          _, preds = torch.max(cpuout, 1) # get predicted class 
          accuracy =  (  accuracy*datasize + torch.sum(preds == labels) ) / ( datasize + inputs.shape[0])
            
          datasize += inputs.shape[0] #update datasize used in accuracy comp
    
    if criterion is None:   
      avgloss = None
          
    return accuracy, avgloss

#function for validation on val data
def train_modelcv(dataloader_cvtrain, dataloader_cvtest ,  model ,  criterion, optimizer, scheduler, num_epochs, device):

  best_measure = 0
  best_epoch =-1

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    losses=train_epoch(model,  dataloader_cvtrain,  criterion,  device , optimizer )
    #scheduler.step()
    measure,_ = evaluate(model, dataloader_cvtest, criterion = None, device = device)
    
    print(' perfmeasure', measure.item() )

    # store current
    if measure > best_measure: 
      bestweights= model.state_dict()
      best_measure = measure
      best_epoch = epoch
      print('current best', measure.item(), ' at epoch ', best_epoch)

  return best_epoch, best_measure, bestweights


def run():

    #set default hp values
    batchsize = 32
    maxnumepochs = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #else "cuda:0" with gpu

    #construct data using dataset class
    ds = {
        'train': EuroSatMsDataset(DATASET_ROOT, PROJECT_ROOT / "data_splits" / DATASET_NAME / "train.csv"),
        'test': EuroSatMsDataset(PROJECT_ROOT, PROJECT_ROOT / "data_splits" / DATASET_NAME / "test.csv"),
        'val': EuroSatMsDataset(PROJECT_ROOT, PROJECT_ROOT / "data_splits" / DATASET_NAME / "val.csv"),
    }

    #load data into batch sizes
    dataloaders = {
        'train':  torch.utils.data.DataLoader(ds['train'], batch_size=batchsize, shuffle=False), 
        'val': torch.utils.data.DataLoader(ds['val'], batch_size=batchsize, shuffle=False), 
        'test':  torch.utils.data.DataLoader(ds['test'], batch_size=batchsize, shuffle=False),  
    }

    print(len(ds['train']))
    num_classes = len(ds['train'].classes)
    loss = torch.nn.CrossEntropyLoss()

    #params for cross validation
    lrates=[0.01, 0.001]
    best_hyperparameter= None
    weights_chosen = None
    bestmeasure = None

    #iterate trough possible hyperparameters
    for lr in lrates: 

        print('\n\n\n###################NEW RUN##################')
        print('############################################')
        print('############################################')

        #reset model for every new hyperparameter selection
        model = EuroSatResNet18(num_classes=num_classes, pretrained=True).to(device)

        #optimizer here because of learning rate
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # train on train and eval on val data 
        best_epoch, best_perfmeasure, bestweights = train_modelcv(dataloader_cvtrain = dataloaders['train'], dataloader_cvtest = dataloaders['val'] ,  model = model ,  criterion = loss , optimizer = optimizer, scheduler = None, num_epochs = maxnumepochs , device = device)

        if best_hyperparameter is None:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure
        elif best_perfmeasure > bestmeasure:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure

    # end of for loop over hyperparameters here!
    model.load_state_dict(weights_chosen)

    accuracy,_ = evaluate(model = model , dataloader  = dataloaders['test'], criterion = None, device = device)

    print('accuracy val',bestmeasure.item() , 'accuracy test',accuracy.item()  )

if __name__=='__main__':
  run()
    



    