"""Reusable preprocessing helpers for the Tiny ImageNet notebook."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedShuffleSplit


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
IMAGES_DIR = DATA_DIR / "images"
SEED = 42


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


def apply_duplicate_review(labels, decisions):
    """Apply reviewed exact/perceptual duplicate exclusions and audit removals."""
    required = {"filename", "decision", "reason"}
    missing_columns = required - set(decisions.columns)
    if missing_columns:
        raise ValueError(f"duplicate decisions are missing columns: {sorted(missing_columns)}")

    decisions = decisions[["filename", "decision", "reason"]].copy()
    decisions["filename"] = decisions["filename"].astype(str).str.strip()
    decisions["decision"] = decisions["decision"].fillna("").astype(str).str.strip().str.lower()
    if decisions["filename"].duplicated().any():
        raise ValueError("duplicate-resolution decisions contain repeated filenames")

    invalid = set(decisions["decision"]) - {"keep", "exclude"}
    if invalid:
        raise ValueError(f"duplicate decisions must be 'keep' or 'exclude'; found {sorted(invalid)}")
    excluded_reasons = (
        decisions.loc[decisions["decision"] == "exclude", "reason"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    if excluded_reasons.eq("").any():
        raise ValueError("provide a reason for every excluded duplicate candidate")

    labels = labels.copy()
    labels["filename"] = labels["filename"].astype(str)
    known_filenames = set(labels["filename"])
    unknown = set(decisions["filename"]) - known_filenames
    if unknown:
        raise ValueError(
            f"duplicate decisions reference unknown filenames: {sorted(unknown)[:10]}"
        )

    conflict_rows, _ = find_duplicate_rows(labels)
    conflicting_filenames = set(conflict_rows["filename"])
    missing_conflicts = conflicting_filenames - set(decisions["filename"])
    if missing_conflicts:
        raise ValueError(
            "duplicate-resolution file is missing exact-content conflict decisions: "
            f"{sorted(missing_conflicts)[:10]}"
        )

    exact_decisions = decisions[decisions["filename"].isin(conflicting_filenames)]
    clean, removed = resolve_duplicate_images(labels, exact_decisions)
    removed_by_filename = {str(filename): reason for filename, reason in removed}
    excluded_candidates = decisions[decisions["decision"] == "exclude"]
    remaining_exclusions = excluded_candidates[
        excluded_candidates["filename"].isin(clean["filename"])
    ]
    for row in remaining_exclusions.itertuples(index=False):
        removed_by_filename[str(row.filename)] = str(row.reason).strip()

    excluded_filenames = set(excluded_candidates["filename"])
    clean = clean.loc[~clean["filename"].isin(excluded_filenames)].reset_index(drop=True)
    return clean, list(removed_by_filename.items())


def remove_suspicious_images(labels, suspicious_filenames):
    """Return a copy of the dataset without filenames flagged as likely mislabels."""
    required = {"filename"}
    missing_columns = required - set(labels.columns)
    if missing_columns:
        raise ValueError(f"labels are missing columns: {sorted(missing_columns)}")

    suspicious = {str(filename).strip() for filename in suspicious_filenames}
    if not suspicious:
        return labels.copy().reset_index(drop=True)

    cleaned = labels.loc[~labels["filename"].astype(str).isin(suspicious)].copy()
    return cleaned.reset_index(drop=True)


def measure_image_sharpness(image):
    """Estimate focus using the variance of the grayscale Laplacian."""
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    if grayscale.ndim != 2 or min(grayscale.shape) < 3:
        raise ValueError("sharpness measurement requires an image at least 3x3 pixels")

    center = grayscale[1:-1, 1:-1]
    laplacian = (
        grayscale[:-2, 1:-1]
        + grayscale[2:, 1:-1]
        + grayscale[1:-1, :-2]
        + grayscale[1:-1, 2:]
        - 4 * center
    )
    return float(np.var(laplacian, dtype=np.float64))


def preprocess_images(labels, source_images_dir, destination_images_dir):
    """Autocontrast images, convert them to RGB, and return sharpness measurements."""
    required = {"filename"}
    missing_columns = required - set(labels.columns)
    if missing_columns:
        raise ValueError(f"labels are missing columns: {sorted(missing_columns)}")

    filenames = labels["filename"].astype(str)
    if filenames.duplicated().any():
        raise ValueError("cannot preprocess images for duplicate filenames")
    invalid_filenames = [
        filename for filename in filenames if Path(filename).name != filename
    ]
    if invalid_filenames:
        raise ValueError(
            f"image filenames must not contain directory paths: {invalid_filenames[:10]}"
        )

    source_images_dir = Path(source_images_dir)
    destination_images_dir = Path(destination_images_dir)
    destination_images_dir.mkdir(parents=True, exist_ok=True)
    sharpness_rows = []

    for filename in filenames:
        source_path = source_images_dir / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"image is missing: {source_path}")
        if source_path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError(f"expected a JPEG image, found: {source_path}")

        with Image.open(source_path) as image:
            sharpness_rows.append({
                "filename": filename,
                "sharpness": measure_image_sharpness(image),
            })
            normalization_input = image if image.mode == "L" else image.convert("RGB")
            processed = ImageOps.autocontrast(normalization_input).convert("RGB")
            processed.save(
                destination_images_dir / filename,
                format="JPEG",
                quality=95,
                subsampling=0,
            )
    return pd.DataFrame(sharpness_rows, columns=["filename", "sharpness"])


def save_cleaned_dataset(labels, output_path):
    """Save a cleaned copy of the dataset and return the output path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.copy().to_csv(output_path, index=False)
    return output_path


def save_split_manifests(train, val, test, classes_table, split_dir):
    """Save the stratified train, validation, test, and class manifests."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    columns = ["filename", "class_id", "description", "label"]
    for filename, frame in (
        ("train.csv", train),
        ("val.csv", val),
        ("test.csv", test),
    ):
        missing_columns = set(columns) - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{filename} data is missing columns: {sorted(missing_columns)}"
            )
        frame[columns].to_csv(split_dir / filename, index=False)
    classes_table.to_csv(split_dir / "classes.csv", index=False)
    return split_dir


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


