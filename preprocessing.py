"""
Preprocessing for the Tiny ImageNet project.
Order: hash images -> remove duplicates -> split -> remove label errors (train only)
-> build loaders -> save csv files.
"""

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE / "Tiny_ImageNet_Project_2026_Students"
IMAGES_DIR = DATA_DIR / "images"
OUT_DIR = BASE / "splits"
OUT_DIR.mkdir(exist_ok=True)

removed = []  # (filename, reason) for every image we drop


# 1. load labels and hash every image
labels = pd.read_csv(DATA_DIR / "labels.csv")
labels["hash"] = [hashlib.md5((IMAGES_DIR / f).read_bytes()).hexdigest()
                  for f in labels["filename"]]


# 2. duplicates (done before the split so no image ends up in two sets)
# same image with different labels -> we can't know which is right, drop all copies
conflict = labels.groupby("hash")["class_id"].transform("nunique") > 1
# same image with the same label -> keep the first one
same_label_copy = labels.duplicated(subset=["hash", "class_id"]) & ~conflict

removed += [(f, "duplicate with conflicting labels") for f in labels.loc[conflict, "filename"]]
removed += [(f, "duplicate with same label") for f in labels.loc[same_label_copy, "filename"]]

clean = labels[~(conflict | same_label_copy)].reset_index(drop=True)
print("removed", len(labels) - len(clean), "images, left:", len(clean))


# 3. class ids -> numbers 0..199
class_ids = sorted(clean["class_id"].unique())
clean["label"] = clean["class_id"].map({c: i for i, c in enumerate(class_ids)})
NUM_CLASSES = len(class_ids)

classes_table = (clean[["label", "class_id", "description"]]
                 .drop_duplicates().sort_values("label"))


# 4. stratified split 70 / 15 / 15
def split(df, test_size):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    a, b = next(sss.split(df, df["label"]))
    return df.iloc[a], df.iloc[b]

train, rest = split(clean, 0.30)
val, test = split(rest, 0.50)
train, val, test = (d.reset_index(drop=True) for d in (train, val, test))

# sanity checks
assert not set(train["hash"]) & set(val["hash"])
assert not set(train["hash"]) & set(test["hash"])
assert not set(val["hash"]) & set(test["hash"])
for name, d in [("train", train), ("val", val), ("test", test)]:
    assert d["label"].nunique() == NUM_CLASSES
    print(name, len(d))


# 5. label errors, train only (val and test stay untouched)
# confirmed_label_errors.csv is filled by hand after looking at the suspicious
# images from the notebook. It needs a "filename" column, "reason" is optional.
CONFIRMED = BASE / "confirmed_label_errors.csv"

if CONFIRMED.exists():
    conf = pd.read_csv(CONFIRMED)
    reasons = dict(zip(conf["filename"], conf["reason"])) if "reason" in conf.columns else {}
    bad = train["filename"].isin(set(conf["filename"]))
    removed += [(f, "label error: " + str(reasons.get(f, "checked by hand")))
                for f in train.loc[bad, "filename"]]
    train = train[~bad].reset_index(drop=True)
    print("removed", bad.sum(), "label errors from train, new size:", len(train))
else:
    print("no confirmed_label_errors.csv, nothing removed")


# 6. transforms, datasets and loaders
MEAN = [0.485, 0.456, 0.406]  # ImageNet values
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
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
        img = Image.open(IMAGES_DIR / self.files[i]).convert("RGB")  # some are grayscale
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


BATCH_SIZE = 128
train_loader = DataLoader(TinyImageNetDataset(train, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(TinyImageNetDataset(val, eval_tf),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(TinyImageNetDataset(test, eval_tf),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# class weights from the train set only
counts = np.bincount(train["label"], minlength=NUM_CLASSES)
class_weights = torch.tensor(len(train) / (NUM_CLASSES * counts), dtype=torch.float32)

imgs, ys = next(iter(train_loader))
print("one batch:", tuple(imgs.shape), tuple(ys.shape))  # (128, 3, 64, 64) (128,)


# 7. save everything
cols = ["filename", "class_id", "description", "label"]
train[cols].to_csv(OUT_DIR / "train.csv", index=False)
val[cols].to_csv(OUT_DIR / "val.csv", index=False)
test[cols].to_csv(OUT_DIR / "test.csv", index=False)
classes_table.to_csv(OUT_DIR / "classes.csv", index=False)
pd.DataFrame(removed, columns=["filename", "reason"]).to_csv(
    BASE / "preprocessing_exclusions.csv", index=False)
print("saved, images removed in total:", len(removed))