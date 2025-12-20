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

    num_classes = len(dataloader.dataset.classes)
    model.eval() # set model to inference mode


    with torch.no_grad(): # do not record computations for computing the gradient
    
        datasize = 0
        accuracy = 0
        avgloss = 0
        logits = []
  
        # create confusion matrix (num_classes, num_classes)
        confusion_matrix = torch.zeros(num_classes, num_classes)
  
        for ctr, data in enumerate(dataloader):
  
            #print ('epoch at',len(dataloader.dataset), ctr)
            
            inputs = data[0].to(device)        
            outputs = model(inputs)
  
            labels = data[1]
  
            # computing some loss
            cpuout = outputs.to('cpu')
            logits.append(cpuout)
            if criterion is not None:
                curloss = criterion(cpuout, labels)
                avgloss = ( avgloss*datasize + curloss ) / ( datasize + inputs.shape[0])
  
            # for computing the accuracy
            labels = labels.float()
            _, preds = torch.max(cpuout, 1) # get predicted class 
            accuracy =  (accuracy*datasize + torch.sum(preds == labels) ) / ( datasize + inputs.shape[0])
  
            # update confusion matrix
            for i in range(num_classes):
                for j in range(num_classes):
                    confusion_matrix[i, j] += torch.sum((labels == i) & (preds == j))
          
            datasize += inputs.shape[0] #update datasize used in accuracy comp
    
    if criterion is None:   
        avgloss = None
    
    # calculate tpr for each class
    tpr_per_class = torch.zeros(num_classes)
    for i in range(num_classes):
        tpr_per_class[i] = confusion_matrix[i, i] / torch.sum(confusion_matrix[i, :])
    
    return accuracy, avgloss, tpr_per_class, torch.cat(logits)

#function for validation on val data
def train_modelcv(dataloader_cvtrain, dataloader_cvtest ,  model ,  criterion, optimizer, aug_name, scheduler, num_epochs, device):

    best_measure = 0
    best_epoch =-1
    bestweights = None
  
    graph_data = []
  
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
    
        losses=train_epoch(model,  dataloader_cvtrain,  criterion,  device , optimizer )
        #scheduler.step()
        measure,_, tpr_per_class,_ = evaluate(model, dataloader_cvtest, criterion = None, device = device)
    
        graph_data.append((epoch, measure.item(), tpr_per_class))
        
        print(' perfmeasure', measure.item() )
    
        # store current
        if measure > best_measure: 
            bestweights= model.state_dict()
            best_measure = measure
            best_epoch = epoch
            print('current best', measure.item(), ' at epoch ', best_epoch)
  
    # plot graph:
    # - epochs on x axis
    # - measure on y axis
    # - tpr per class on y axis
    epochs = [x[0] for x in graph_data]
    measures = [x[1] for x in graph_data]
    tprs = torch.stack([x[2] for x in graph_data]).cpu().numpy()
  
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, measures, label='Accuracy', linewidth=2, color='black')
    
    class_names = dataloader_cvtrain.dataset.classes
    for i, cls_name in enumerate(class_names):
        plt.plot(epochs, tprs[:, i], label=f'TPR {cls_name}', linestyle='--')
  
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title(f'Training Metrics (LR={optimizer.param_groups[0]["lr"]}, Augmentation={aug_name})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(results_dir / 'plots', exist_ok=True)
    plt.savefig(results_dir / f'plots/training_plot_lr_{optimizer.param_groups[0]["lr"]}_{aug_name}.png')
    
    return best_epoch, best_measure, bestweights


def run():

    #set default hp values
    batchsize = 32
    maxnumepochs = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #else "cuda:0" with gpu

    # Define augmentation settings
    #aug_settings = {
    #    'Strong Augmentation': transforms.Compose([
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #        transforms.RandomRotation(10)
    #    ]),
    #    'Mild Augmentation': transforms.Compose([
    #        transforms.RandomHorizontalFlip()
    #    ]),
    #    'No Augmentation': None,
    #}
    aug_settings = {
        'No Augmentation': None,
    }

    #params for cross validation
    #lrates=[0.001, 0.01, 0.0001]
    lrates=[0.001]

    best_augmentation = None
    best_hyperparameter= None
    weights_chosen = None
    bestmeasure = 0.0
    
    # Iterate over augmentation settings
    for aug_name, aug_transform in aug_settings.items():
        print(f"\n\n\n============================================")
        print(f"Running Experiment with: {aug_name}")
        print(f"============================================")

        #construct data using dataset class
        ds = {
            'train': EuroSatMsDataset(DATASET_ROOT, PROJECT_ROOT / "data_splits" / DATASET_NAME / "train.csv", transform=aug_transform),
            'test': EuroSatMsDataset(DATASET_ROOT, PROJECT_ROOT / "data_splits" / DATASET_NAME / "test.csv"),
            'val': EuroSatMsDataset(DATASET_ROOT, PROJECT_ROOT / "data_splits" / DATASET_NAME / "val.csv"),
        }

        #load data into batch sizes
        dataloaders = {
            'train':  torch.utils.data.DataLoader(ds['train'], batch_size=batchsize, shuffle=True), 
            'val': torch.utils.data.DataLoader(ds['val'], batch_size=batchsize, shuffle=False), 
            'test':  torch.utils.data.DataLoader(ds['test'], batch_size=batchsize, shuffle=False),  
        }

        num_classes = len(ds['train'].classes)
        loss = torch.nn.CrossEntropyLoss()

        #iterate trough possible hyperparameters
        for lr in lrates: 

            print('\n--------------------------------------------')
            print(f'Hyperparameter search: LR={lr}')
            print('--------------------------------------------')

            #reset model for every new hyperparameter selection
            model = EuroSatResNet18(num_classes=num_classes).to(device)

            #optimizer here because of learning rate
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # train on train and eval on val data 
            best_epoch, best_perfmeasure, bestweights = train_modelcv(dataloader_cvtrain = dataloaders['train'], dataloader_cvtest = dataloaders['val'] ,  model = model ,  criterion = loss , optimizer = optimizer, aug_name = aug_name, scheduler = None, num_epochs = maxnumepochs , device = device)

            if best_hyperparameter is None:
                best_augmentation = aug_name
                best_hyperparameter = lr
                weights_chosen = bestweights
                bestmeasure = best_perfmeasure
            elif best_perfmeasure > bestmeasure:
                best_augmentation = aug_name
                best_hyperparameter = lr
                weights_chosen = bestweights
                bestmeasure = best_perfmeasure

    # end of for loop over augmentations/hyperparameters here!
    model.load_state_dict(weights_chosen)
    accuracy,_,tpr_per_class,logits = evaluate(model = model , dataloader  = dataloaders['test'], criterion = None, device = device)
    report = f'Final Result: val_acc={bestmeasure.item():.4f}, test_acc={accuracy.item():.4f}\n'

    class_names = dataloaders['train'].dataset.classes
    for i, cls_name in enumerate(class_names):
        report += f'{cls_name:<22}: {tpr_per_class[i]:.4f}\n'
    
    report += f'\nBest Hyperparameter: LR={best_hyperparameter}\n'
    report += f'Best Augmentation: {best_augmentation}\n'
    
    report += f'Num Epochs: {maxnumepochs}\n'
    report += f'Batch Size: {batchsize}\n'
    report += f'All Hyperparameters: LR={lrates}\n'
    report += f'All Augmentations: {aug_settings.keys()}\n'
    print(report)

    with open(results_dir / 'report.txt', 'w') as f:
        f.write(report)
    
    with open(results_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    torch.save(logits, results_dir / 'logits.pt')
    pd.DataFrame({'image_path': dataloaders['test'].dataset.images.iloc[:, 0]}).to_csv(results_dir / 'logits.csv', index=False)

if __name__=='__main__':
  run()
    