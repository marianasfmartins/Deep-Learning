"""
preprocessing.py - Tiny ImageNet project (data preparation)

Steps (same order as the conclusions of the EDA):
  1. load labels.csv and hash every image
  2. remove duplicate images (BEFORE splitting)
  3. class ids -> numbers 0..199
  4. stratified split 70 / 15 / 15 (train / val / test)
  5. remove confirmed label errors (TRAIN only)
  6. PyTorch datasets, loaders and class weights
  7. save everything so the group can just load the files

Rule from class: split first, and anything that learns from the data
(like the class weights) is computed on TRAIN only -> no data leakage.
"""

# %% Imports and settings
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# one seed for everything, so the results are the same every time we run it
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# paths are built from the location of this file (in a notebook we use the current folder)
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE / "Tiny_ImageNet_Project_2026_Students"
IMAGES_DIR = DATA_DIR / "images"
OUT_DIR = BASE / "splits"
OUT_DIR.mkdir(exist_ok=True)


# %% 1. Load labels and hash the images
labels = pd.read_csv(DATA_DIR / "labels.csv")   # we only read this file, never overwrite it
print("rows in labels.csv:", len(labels))

# identical images have the same hash, so the hash lets us find duplicates
labels["hash"] = [hashlib.md5((IMAGES_DIR / f).read_bytes()).hexdigest()
                  for f in labels["filename"]]


# %% 2. Remove duplicates (before splitting!)
# If a copy of an image ended up in train and another in test, the test score would be
# too optimistic (data leakage). So we clean duplicates first.

# a) same image, different labels -> we can't know which label is right, drop ALL copies
n_classes_per_hash = labels.groupby("hash")["class_id"].transform("nunique")
conflict = n_classes_per_hash > 1

# b) same image, same label -> harmless but useless, keep only the first copy
redundant = labels.duplicated(subset=["hash", "class_id"]) & ~conflict

# keep a log of what we removed, useful for the report
exclusions = pd.concat([
    labels[conflict].assign(reason="conflicting_duplicate"),
    labels[redundant].assign(reason="redundant_duplicate"),
])[["filename", "class_id", "reason"]]

clean = labels[~(conflict | redundant)].reset_index(drop=True)
print(f"removed {len(exclusions)} images | remaining {len(clean):,}")
assert not clean["hash"].duplicated().any()      # check: every image is unique now


# %% 3. Class ids -> numbers
# the network needs numbers (0..199), not strings like "n01443537"
class_ids = sorted(clean["class_id"].unique())
class_to_idx = {c: i for i, c in enumerate(class_ids)}
clean["label"] = clean["class_id"].map(class_to_idx)
NUM_CLASSES = len(class_ids)
print("number of classes:", NUM_CLASSES)

# lookup table (label -> class_id -> description), needed for the error analysis
classes_table = (clean[["label", "class_id", "description"]]
                 .drop_duplicates().sort_values("label"))


# %% 4. Split 70 / 15 / 15
# stratify = every split keeps the same class proportions
# first cut: 70% train | 30% rest. second cut: the rest is divided in half -> val and test
train, rest = train_test_split(clean, test_size=0.30, stratify=clean["label"],
                               random_state=SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"],
                             random_state=SEED)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

# checks: all classes in every split, and no image shared between splits
for name, part in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:5s} {len(part):6,d} images | {part['label'].nunique()} classes")
    assert part["label"].nunique() == NUM_CLASSES

for a, b in [(train, val), (train, test), (val, test)]:
    assert not set(a["hash"]) & set(b["hash"])
print("check ok: no overlap between the splits")


# %% 5. Label errors (TRAIN only)
# high_priority_suspicious_labels.csv comes from the EDA notebook (section 1.10.5)
REGISTRY = BASE / "high_priority_suspicious_labels.csv"
# confirmed_label_errors.csv is made BY HAND after looking at the images (one column: filename).
# Only images we are sure are wrong go in there, the unsure ones stay in the data.
CONFIRMED = BASE / "confirmed_label_errors.csv"

# a) sheet with the suspicious train images, to review them by hand
if REGISTRY.exists():
    registry = pd.read_csv(REGISTRY)
    review = registry[registry["filename"].isin(train["filename"])]
    review = review.sort_values(["inconsistency_score", "consensus_ratio"], ascending=False)
    review.to_csv(OUT_DIR / "train_noise_review.csv", index=False)
    print(f"{len(review)} of {len(registry)} suspicious images are in train "
          f"-> splits/train_noise_review.csv")
else:
    print("no registry file found -> skipping the review sheet")

# b) remove the confirmed ones
if CONFIRMED.exists():
    confirmed = set(pd.read_csv(CONFIRMED)["filename"])
    bad = train["filename"].isin(confirmed)
    # val and test are NEVER changed (even if a confirmed image is there),
    # so the final score stays honest
    print(f"confirmed: {len(confirmed)} | removed from train: {bad.sum()} | "
          f"ignored (val/test): {len(confirmed) - bad.sum()}")
    exclusions = pd.concat([
        exclusions,
        train[bad][["filename", "class_id"]].assign(reason="confirmed_label_error"),
    ])
    train = train[~bad].reset_index(drop=True)
else:
    print("no confirmed_label_errors.csv yet -> nothing removed")

print(f"final sizes -> train {len(train):,} | val {len(val):,} | test {len(test):,}")


# %% 6. PyTorch pipeline
# ImageNet mean/std because the pre-trained models expect it.
# These numbers are fixed (not learned from our data), so there is no leakage.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# train: light augmentation (the images are only 64x64, so we keep it mild)
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# val / test: no augmentation, otherwise the evaluation would be random
# (for transfer learning we can add transforms.Resize(...) at the start of both)
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class TinyImageNetDataset(Dataset):
    def __init__(self, df, transform=None):
        self.files = df["filename"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # convert("RGB") for every image, because some of them are grayscale (1 channel)
        img = Image.open(IMAGES_DIR / self.files[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


BATCH_SIZE = 128
# num_workers=0 because multiprocessing is slow to start (or crashes) on Windows
train_loader = DataLoader(TinyImageNetDataset(train, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(TinyImageNetDataset(val, eval_tf),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(TinyImageNetDataset(test, eval_tf),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# class weights for the loss: small classes get a bigger weight
# weight = n_samples / (n_classes * n_samples_of_that_class)
# computed on train only, AFTER the label errors were removed
counts = np.bincount(train["label"], minlength=NUM_CLASSES)
class_weights = torch.tensor(len(train) / (NUM_CLASSES * counts), dtype=torch.float32)
print(f"images per class in train: min {counts.min()} | max {counts.max()} "
      f"| weights from {class_weights.min():.2f} to {class_weights.max():.2f}")

# check one batch, expected: (128, 3, 64, 64) and (128,)
imgs, ys = next(iter(train_loader))
print("batch:", tuple(imgs.shape), tuple(ys.shape), "| dtype:", imgs.dtype)


# %% 7. Save the results
cols = ["filename", "class_id", "description", "label"]
train[cols].to_csv(OUT_DIR / "train.csv", index=False)
val[cols].to_csv(OUT_DIR / "val.csv", index=False)
test[cols].to_csv(OUT_DIR / "test.csv", index=False)
classes_table.to_csv(OUT_DIR / "classes.csv", index=False)
exclusions.to_csv(OUT_DIR / "preprocessing_exclusions.csv", index=False)
np.save(OUT_DIR / "class_weights.npy", class_weights.numpy())   # so the others don't recompute it
print("saved to", OUT_DIR)