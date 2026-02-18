import random
from pathlib import Path
from PIL import Image

import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# CONFIG
DATASET_PATH = Path("/Volumes/lhe-drive-1/datasets/102flowers/flowers_data")
SEED = 20260218
TORCH_DEVICE = "mps"
LIMIT_DATA = 500
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_CLASSES = 102

# SEED
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


# DATASET
class Flowsers102Dataset(Dataset):
    def __init__(self, split_name, transform):
        df = pd.read_csv(DATASET_PATH / f"{split_name}file.txt", sep=" ")
        if LIMIT_DATA:
            df = df.head(LIMIT_DATA)
        self.img_names = df.iloc[:, 0].tolist()
        self.labels = df.iloc[:, 1].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        img_path = DATASET_PATH / "jpg" / self.img_names[idx]
        image = Image.open(img_path)
        image = self.transform(image)

        return image, label


# TRAINING
def run():
    # Load dataset
    data_transforms = get_data_augmentation()
    datasets = get_datasets(data_transforms)
    dataloader = get_data_loader(datasets)

    # Load model
    device = torch.device(TORCH_DEVICE)
    model = load_model()
    model.to(device)

    # Init training
    criterion = torch.nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

    # Run training
    _, _, best_weights = train_model(device, dataloader, model, criterion, optimizer)

    # Evaluate on the test set
    model.load_state_dict(best_weights)
    measure = evaluate(device, dataloader["test"], model)
    print(f"Accuracy on test set: {measure.item()}")


def get_data_augmentation():
    return {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # further augmentations here or after the totensor
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def get_datasets(data_transforms):
    return {
        "train": Flowsers102Dataset("train", data_transforms["train"]),
        "val": Flowsers102Dataset("val", data_transforms["val"]),
        "test": Flowsers102Dataset("test", data_transforms["test"]),
    }


def get_data_loader(datasets):
    return {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        ),
    }


def load_model():
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Adapt classifier to predict 102 classes (automatically unfrozen)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    # Also unfreeze second last layer
    for param in model.features[7].parameters():
        param.requires_grad = True

    return model


def train_model(device, dataloader, model, criterion, optimizer):
    best_epoch = -1
    best_measure = -1
    best_weights = None

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}/{NUM_EPOCHS-1}")
        print("=" * 10)

        train_epoch(device, dataloader["train"], model, criterion, optimizer)
        measure = evaluate(device, dataloader["val"], model)

        print(f"Accuracy: {measure.item()}")

        if measure > best_measure:
            best_epoch = epoch
            best_measure = measure
            best_weights = model.state_dict()

        print(f"Current best is {measure.item()} at epoch {best_epoch}\n")

    return best_epoch, best_measure, best_weights


def train_epoch(device, train_dataloader, model, criterion, optimizer):
    model.train()

    for batch_idx, data in enumerate(train_dataloader):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # reset accumulated gradients
        loss.backward()  # compute new gradients
        optimizer.step()  # apply new gradients to change model parameters


def evaluate(device, eval_dataloader, model):
    model.eval()

    with torch.no_grad():
        datasize = 0
        accuracy = 0

        for ctr, data in enumerate(eval_dataloader):
            inputs = data[0].to(device)
            labels = data[1].float()

            outputs = model(inputs)
            cpuout = outputs.to("cpu")

            # compute the accuracy batch-wise
            _, preds = torch.max(cpuout, 1)  # get predicted class
            accuracy = (accuracy * datasize + torch.sum(preds == labels)) / (
                datasize + inputs.shape[0]
            )
            datasize += inputs.shape[0]  # update datasize used in accuracy computation

    return accuracy


if __name__ == "__main__":
    run()
