from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import csv
import os

# set root path and seed
PROJECT_ROOT = Path(os.getcwd()) #Path(r"C:\Users\tdoro\DLMS\mandatory_task")
DATASET_NAME = "EuroSAT_MS"
DATASET_ROOT = PROJECT_ROOT / DATASET_NAME

# use mat-nr as seed
RANDOM_SEED = 3778660
random.seed(RANDOM_SEED)

train_list = []
val_list = []
test_list = []

#iterate trough dataset dirs/class folders and perform train/val/test splits on every class seperately
for class_dir in sorted(DATASET_ROOT.iterdir()):
    if class_dir.is_dir():
        label = class_dir.name

        class_files = [p.relative_to(PROJECT_ROOT) for p in class_dir.glob("*.tif")]
        
        #splitting data into train, val, test
        train, test = train_test_split(class_files, test_size=0.2, random_state = RANDOM_SEED)
        train, val = train_test_split(train, test_size = 0.2, random_state = RANDOM_SEED)

        #write into lists, adding tuple with filepath and label
        train_list.extend((str(p), label) for p in train)
        val_list.extend((str(p), label) for p in val)
        test_list.extend((str(p), label) for p in test)

#check length
print(len(train_list))
print(len(val_list))
print(len(test_list))

#check if disjoint -> no error output when disjoint. 
#have to turn lists into sets first to use isdisjoint operator
assert set(train_list).isdisjoint(set(val_list)) 
assert set(train_list).isdisjoint(set(test_list))
assert set(val_list).isdisjoint(set(test_list))

#shuffle lists just to be sure
random.shuffle(train_list)
random.shuffle(val_list)
random.shuffle(test_list)

#select smaller amount of datapoints due to hardware constraints
train_list = train_list[:2500]
val_list = val_list[:1000]
test_list = test_list[:2000]

#check length
print(len(train_list))
print(len(val_list))
print(len(test_list))

#save data in 3 seperate csv files
# 2 columns with "filepath, label" as per tupels stored in lists
def write_csv(path, data):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])  # Header
        for fp, label in data:
            writer.writerow([fp, label])

write_csv(PROJECT_ROOT / "train.csv", train_list)
write_csv(PROJECT_ROOT / "val.csv",   val_list)
write_csv(PROJECT_ROOT / "test.csv",  test_list)
