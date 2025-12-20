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

# use mat-nr as seed
RANDOM_SEED = 3778660
random.seed(RANDOM_SEED)

class EuroSatMsDataset(Dataset):
    def __init__(self, dataset_root_dir, split_name):
        self.dataset_root_dir = dataset_root_dir
        self.img_labels = pd.read_csv(self.dataset_root_dir / "EuroSAT_MS" / (split_name + ".csv"))

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

ds_train = EuroSatMsDataset(PROJECT_ROOT, "train")
ds_test = EuroSatMsDataset(PROJECT_ROOT, "test")
ds_val = EuroSatMsDataset(PROJECT_ROOT, "val")

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False)
val_loader = torch.utils.data.DataLoader(ds_val, batch_size=32, shuffle=False)

# Define the model
model = resnet18(pretrained=True)

# Replace the last layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(ds_train.classes))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move the model to the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the number of epochs
num_epochs = 10

# Train the model
for epoch in range(num_epochs):
    # Train the model on the training set
    model.train()
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Move the data to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update the training loss
        train_loss += loss.item() * inputs.size(0)

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            # Move the data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update the test loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == labels.data)

    # Print the training and test loss and accuracy
    train_loss /= len(ds_train)
    test_loss /= len(ds_test)
    test_acc = test_acc.double() / len(ds_test)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")


