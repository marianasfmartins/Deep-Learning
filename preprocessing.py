"""Reusable preprocessing helpers for the Tiny ImageNet notebook."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
IMAGES_DIR = DATA_DIR / "images"
SEED = 42
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_and_hash_labels(data_dir=DATA_DIR):
    """Load the manifest and attach an MD5 hash to each image's contents."""
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels = pd.read_csv(data_dir / "labels.csv")
    required = {"filename", "class_id", "description"}
    missing_columns = required - set(labels.columns)
    if missing_columns:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing_columns)}")
    if labels["filename"].duplicated().any():
        duplicates = labels.loc[labels["filename"].duplicated(False), "filename"].tolist()
        raise ValueError(f"labels.csv contains duplicate filenames: {duplicates[:10]}")

    missing_files = [
        filename for filename in labels["filename"]
        if not (images_dir / str(filename)).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} manifest images are missing from {images_dir}; "
            f"examples: {missing_files[:10]}"
        )

    labels = labels.copy()
    labels["filename"] = labels["filename"].astype(str)
    labels["hash"] = [
        hashlib.md5((images_dir / filename).read_bytes()).hexdigest()
        for filename in labels["filename"]
    ]
    return labels


def find_duplicate_rows(labels):
    """Return conflicting duplicate rows and redundant same-label duplicate rows."""
    required = {"filename", "class_id", "hash"}
    missing_columns = required - set(labels.columns)
    if missing_columns:
        raise ValueError(f"labels are missing columns: {sorted(missing_columns)}")

    conflicting = labels.groupby("hash")["class_id"].transform("nunique") > 1
    conflict_rows = labels.loc[conflicting].copy().sort_values(["hash", "filename"])
    same_label_copies = labels.duplicated(subset=["hash", "class_id"]) & ~conflicting
    same_label_rows = labels.loc[same_label_copies].copy()
    return conflict_rows, same_label_rows


def write_duplicate_review_template(conflict_rows, review_path):
    """Create a review CSV once, without overwriting prior human decisions."""
    review_path = Path(review_path)
    if conflict_rows.empty or review_path.exists():
        return review_path

    review_path.parent.mkdir(parents=True, exist_ok=True)
    template = conflict_rows[["filename", "class_id", "description", "hash"]].copy()
    template["decision"] = ""
    template["reason"] = ""
    template.to_csv(review_path, index=False)
    return review_path


def resolve_duplicate_images(labels, decisions):
    """Apply human decisions to conflicting copies and keep one same-label copy."""
    conflict_rows, same_label_rows = find_duplicate_rows(labels)
    required = {"filename", "decision", "reason"}
    missing_columns = required - set(decisions.columns)
    if missing_columns:
        raise ValueError(f"duplicate decisions are missing columns: {sorted(missing_columns)}")

    decisions = decisions[["filename", "decision", "reason"]].copy()
    decisions["filename"] = decisions["filename"].astype(str)
    if decisions["filename"].duplicated().any():
        raise ValueError("duplicate-resolution decisions contain repeated filenames")

    expected = set(conflict_rows["filename"])
    supplied = set(decisions["filename"])
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(
            "duplicate-resolution file must contain exactly the conflicting filenames; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    decisions["decision"] = decisions["decision"].fillna("").astype(str).str.strip().str.lower()
    invalid = set(decisions["decision"]) - {"keep", "exclude"}
    if invalid:
        raise ValueError(f"duplicate decisions must be 'keep' or 'exclude'; found {sorted(invalid)}")

    conflict_decisions = conflict_rows.merge(
        decisions, on="filename", how="left", validate="one_to_one"
    )
    excluded_conflicts = []
    removed = []
    for _, group in conflict_decisions.groupby("hash", sort=False):
        keep_count = int((group["decision"] == "keep").sum())
        if keep_count > 1:
            raise ValueError(
                f"conflicting duplicate group {group['hash'].iloc[0]} has more than one kept image"
            )
        excluded = group[group["decision"] == "exclude"]
        for _, row in excluded.iterrows():
            reason = "" if pd.isna(row["reason"]) else str(row["reason"]).strip()
            if not reason:
                raise ValueError(
                    f"provide a reason for excluding duplicate {row['filename']}"
                )
            excluded_conflicts.append(row["filename"])
            removed.append(
                (row["filename"], f"conflicting identical-image labels: {reason}")
            )

    conflict_mask = labels.groupby("hash")["class_id"].transform("nunique") > 1
    same_label_copy_mask = labels.duplicated(subset=["hash", "class_id"]) & ~conflict_mask
    for filename in labels.loc[same_label_copy_mask, "filename"]:
        removed.append((filename, "identical image with the same label; kept first copy"))

    excluded_names = set(excluded_conflicts)
    remove_mask = labels["filename"].isin(excluded_names) | same_label_copy_mask
    clean = labels.loc[~remove_mask].reset_index(drop=True)
    if clean["hash"].duplicated().any():
        raise AssertionError("duplicate image content remains after duplicate resolution")
    return clean, removed


def stratified_split(labels, seed=SEED):
    """Resolve class IDs to integers and split the cleaned data 70/15/15."""
    if labels.empty:
        raise ValueError("cannot split an empty dataset")
    class_ids = sorted(labels["class_id"].unique())
    if len(class_ids) != 200:
        raise ValueError(f"expected 200 classes after duplicate resolution, found {len(class_ids)}")

    labels = labels.copy()
    labels["label"] = labels["class_id"].map({class_id: i for i, class_id in enumerate(class_ids)})
    class_descriptions = labels.groupby("class_id")["description"].nunique()
    if (class_descriptions > 1).any():
        raise ValueError("a class_id maps to multiple descriptions")
    classes_table = (
        labels[["label", "class_id", "description"]]
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
    )

    def split(frame, test_size):
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        first, second = next(splitter.split(frame, frame["label"]))
        return frame.iloc[first], frame.iloc[second]

    train, remainder = split(labels, 0.30)
    val, test = split(remainder, 0.50)
    train, val, test = (
        frame.reset_index(drop=True) for frame in (train, val, test)
    )

    for name, frame in (("train", train), ("validation", val), ("test", test)):
        if frame["label"].nunique() != len(class_ids):
            raise ValueError(f"{name} split does not contain all 200 classes")
    for left, right in ((train, val), (train, test), (val, test)):
        if set(left["hash"]) & set(right["hash"]):
            raise AssertionError("identical image content appears in multiple splits")
    return train, val, test, classes_table


def compute_train_only_noise_audit(train, train_embeddings, k=15):
    """Score each training image using only the other training images as neighbors."""
    if len(train) <= k:
        raise ValueError(f"need more than {k} training images to compute neighbors")
    if len(train) != len(train_embeddings):
        raise ValueError("training rows and training embeddings must have equal lengths")
    if not {"filename", "class_id", "description"}.issubset(train.columns):
        raise ValueError("training data must contain filename, class_id, and description")

    embeddings = np.asarray(train_embeddings)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("training embeddings must be a finite two-dimensional array")

    neighbor_model = NearestNeighbors(
        n_neighbors=k + 1, metric="cosine", algorithm="brute"
    )
    neighbor_model.fit(embeddings)
    neighbor_indices = neighbor_model.kneighbors(
        embeddings, return_distance=False
    )
    neighbors = np.asarray([
        row[row != index][:k] for index, row in enumerate(neighbor_indices)
    ])
    if neighbors.shape != (len(train), k):
        raise ValueError("could not find the requested number of non-self neighbors")

    classes = train["class_id"].to_numpy()
    descriptions = train["description"].to_numpy()
    scores = []
    suggested_classes = []
    suggested_descriptions = []
    consensus_ratios = []

    for index, neighbor_row in enumerate(neighbors):
        neighbor_classes = classes[neighbor_row]
        values, counts = np.unique(neighbor_classes, return_counts=True)
        top_class = values[np.argmax(counts)]
        top_description = descriptions[neighbor_row][
            np.flatnonzero(neighbor_classes == top_class)[0]
        ]
        scores.append(1.0 - np.mean(neighbor_classes == classes[index]))
        suggested_classes.append(top_class)
        suggested_descriptions.append(top_description)
        consensus_ratios.append(counts.max() / k)

    audit = train.copy().reset_index(drop=True)
    audit["inconsistency_score"] = scores
    audit["suggested_class"] = suggested_classes
    audit["suggested_desc"] = suggested_descriptions
    audit["consensus_ratio"] = consensus_ratios
    return audit, neighbors


def create_suggested_training_copy(
    train, candidates, classes_table, exception_filename
):
    """Create a separate training manifest with neighbor-suggested labels."""
    required = {"filename", "suggested_class"}
    missing_columns = required - set(candidates.columns)
    if missing_columns:
        raise ValueError(f"candidates are missing columns: {sorted(missing_columns)}")
    if candidates["filename"].duplicated().any():
        raise ValueError("training candidates contain repeated filenames")

    train_by_filename = train.set_index("filename", drop=False)
    candidate_filenames = set(candidates["filename"].astype(str))
    train_filenames = set(train["filename"].astype(str))
    unexpected = candidate_filenames - train_filenames
    if unexpected:
        raise ValueError(
            f"neighbor candidates must be training-only; examples: {sorted(unexpected)[:10]}"
        )

    class_ids = set(classes_table["class_id"].astype(str))
    suggestions = candidates.copy()
    suggestions["suggested_class"] = suggestions["suggested_class"].astype(str)
    invalid_classes = set(suggestions["suggested_class"]) - class_ids
    if invalid_classes:
        raise ValueError(
            f"neighbor suggestions reference unknown classes: {sorted(invalid_classes)}"
        )

    description_by_class = dict(zip(classes_table["class_id"], classes_table["description"]))
    label_by_class = dict(zip(classes_table["class_id"], classes_table["label"]))
    suggested_train = train.copy()
    audit_rows = []

    for row in suggestions.itertuples(index=False):
        filename = str(row.filename)
        original = train_by_filename.loc[filename]
        if filename == exception_filename:
            applied_class = str(original["class_id"])
            action = "kept original label by exception"
        else:
            applied_class = str(row.suggested_class)
            action = "applied training-only neighbor suggestion"

        applied_description = description_by_class[applied_class]
        mask = suggested_train["filename"].eq(filename)
        suggested_train.loc[mask, "class_id"] = applied_class
        suggested_train.loc[mask, "description"] = applied_description
        suggested_train.loc[mask, "label"] = label_by_class[applied_class]

        audit_rows.append({
            "filename": filename,
            "original_class_id": original["class_id"],
            "original_description": original["description"],
            "suggested_class_id": row.suggested_class,
            "suggested_description": description_by_class[row.suggested_class],
            "applied_class_id": applied_class,
            "applied_description": applied_description,
            "inconsistency_score": getattr(row, "inconsistency_score", np.nan),
            "consensus_ratio": getattr(row, "consensus_ratio", np.nan),
            "action": action,
        })

    if exception_filename in train_filenames and exception_filename not in candidate_filenames:
        original = train_by_filename.loc[exception_filename]
        audit_rows.append({
            "filename": exception_filename,
            "original_class_id": original["class_id"],
            "original_description": original["description"],
            "suggested_class_id": "",
            "suggested_description": "",
            "applied_class_id": original["class_id"],
            "applied_description": original["description"],
            "inconsistency_score": np.nan,
            "consensus_ratio": np.nan,
            "action": "not flagged; original label retained",
        })

    audit_columns = [
        "filename", "original_class_id", "original_description",
        "suggested_class_id", "suggested_description", "applied_class_id",
        "applied_description", "inconsistency_score", "consensus_ratio", "action",
    ]
    relabel_audit = pd.DataFrame(audit_rows, columns=audit_columns)
    return suggested_train, relabel_audit


class TinyImageNetDataset(Dataset):
    def __init__(self, frame, images_dir=IMAGES_DIR, transform=None):
        self.files = frame["filename"].tolist()
        self.labels = frame["label"].tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.images_dir / self.files[index]) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]


def build_dataloaders(train, val, test, images_dir=IMAGES_DIR, batch_size=128):
    """Create augmented training and deterministic validation/test loaders."""
    torch.manual_seed(SEED)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_loader = DataLoader(
        TinyImageNetDataset(train, images_dir, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TinyImageNetDataset(val, images_dir, eval_transform),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        TinyImageNetDataset(test, images_dir, eval_transform),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    counts = np.bincount(train["label"], minlength=200)
    if (counts == 0).any():
        raise ValueError("the reviewed training split has a class with no samples")
    class_weights = torch.tensor(
        len(train) / (len(counts) * counts), dtype=torch.float32
    )
    return train_loader, val_loader, test_loader, class_weights


def save_comparison_outputs(
    train_original, train_suggested, val, test, classes_table,
    relabel_audit, duplicate_exclusions, out_dir=BASE / "splits",
    exclusions_path=BASE / "preprocessing_exclusions.csv",
):
    """Save original and suggested training manifests with shared evaluation splits."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["filename", "class_id", "description", "label"]
    train_original[columns].to_csv(out_dir / "train_original.csv", index=False)
    train_suggested[columns].to_csv(out_dir / "train_suggested.csv", index=False)
    val[columns].to_csv(out_dir / "val.csv", index=False)
    test[columns].to_csv(out_dir / "test.csv", index=False)
    classes_table.to_csv(out_dir / "classes.csv", index=False)
    relabel_audit.to_csv(out_dir / "relabel_audit.csv", index=False)
    pd.DataFrame(duplicate_exclusions, columns=["filename", "reason"]).to_csv(
        exclusions_path, index=False
    )
