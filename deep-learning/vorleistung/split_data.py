import config
import os
import random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASET_NAME = "EuroSAT_MS"
dataset_path = Path(config.DATASET_PATH) / DATASET_NAME

random.seed(config.SEED)

# Find images in data
total_images = [p.relative_to(dataset_path) for p in (dataset_path).glob("*/*.tif")]
total_labels = [img.parent.name for img in total_images]

# Split the data
train_images, test_images, train_labels, test_labels = train_test_split(
    total_images,
    total_labels,
    test_size=config.TEST_SIZE,
    train_size=config.TRAIN_SIZE+config.VAL_SIZE,
    random_state=config.SEED,
    stratify=total_labels
)

train_images, val_images, train_labels, val_labels = train_test_split(
    train_images,
    train_labels,
    test_size=config.VAL_SIZE,
    train_size=config.TRAIN_SIZE,
    random_state=config.SEED,
    stratify=train_labels
)

data_splits = {
    "Total": (total_images, total_labels),
    "Train": (train_images, train_labels),
    "Test": (test_images, test_labels),
    "Val": (val_images, val_labels)
}

# Print stats and checks
print("------------------- Dataset Size -------------------")
for name, (_, labels) in data_splits.items():
    print(f"{name:<10} {len(labels):>8}")
print("----------------------------------------------------\n")


print("---------------- Class Distribution ----------------")
stats = pd.concat(
    [pd.Series(v[1]).value_counts(normalize=True).rename(k) for k, v in data_splits.items()],
    axis=1
)

print(stats.sort_index().map(lambda x: f"{x:.2%}") )
print("----------------------------------------------------\n")


print("------------------ Verify Disjoint -----------------")
train_test_disjoint = not any(img in test_images for img in train_images)
train_val_disjoint = not any(img in val_images for img in train_images)
test_val_disjoint = not any(img in val_images for img in test_images)

print(f"Train and Test sets disjoint = {train_test_disjoint}")
print(f"Train and Val sets disjoint  = {train_val_disjoint}")
print(f"Test and Val sets disjoint   = {test_val_disjoint}")

assert train_test_disjoint
assert train_val_disjoint
assert test_val_disjoint
print("----------------------------------------------------")

# Write to file
os.makedirs(f"{config.PROJECT_ROOT}/data_splits/{DATASET_NAME}", exist_ok=True)

for name, (images, labels) in data_splits.items():
    if name == "Total":
        continue

    df = pd.DataFrame({"filepath": images, "label": labels})
    df["filepath"] = DATASET_NAME + "/" + df["filepath"].astype(str)
    df.to_csv(f"{config.PROJECT_ROOT}/data_splits/{DATASET_NAME}/{name.lower()}.csv", index=False)
