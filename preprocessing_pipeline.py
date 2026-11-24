"""Run the Tiny ImageNet cleaning, image preprocessing, and split pipeline."""

from pathlib import Path

import pandas as pd

from preprocessing_utils import (
    BASE,
    apply_duplicate_review,
    load_and_hash_labels,
    preprocess_images,
    remove_suspicious_images,
    save_cleaned_dataset,
    save_split_manifests,
    stratified_split,
)


def run_preprocessing_pipeline(
    data_dir=BASE / "data",
    duplicate_decisions_path=BASE / "duplicate_resolution.csv",
    suspicious_labels_path=BASE / "high_priority_suspicious_labels.csv",
    output_dir=BASE / "preprocessed_data",
    split_dir=BASE / "splits",
    exclusions_path=BASE / "preprocessing_exclusions.csv",
    minimum_sharpness=10.0,
):
    """Create filtered manifests, normalized RGB images, and stratified splits."""
    data_dir = Path(data_dir)
    duplicate_decisions_path = Path(duplicate_decisions_path)
    suspicious_labels_path = Path(suspicious_labels_path)
    output_dir = Path(output_dir)
    split_dir = Path(split_dir)
    exclusions_path = Path(exclusions_path)

    labels = load_and_hash_labels(data_dir)
    duplicate_decisions = pd.read_csv(duplicate_decisions_path)
    duplicate_clean, duplicate_exclusions = apply_duplicate_review(
        labels, duplicate_decisions
    )

    suspicious_audit = pd.read_csv(suspicious_labels_path)
    if "filename" not in suspicious_audit.columns:
        raise ValueError(
            f"{suspicious_labels_path} is missing the required 'filename' column"
        )
    suspicious_filenames = suspicious_audit["filename"].astype(str).str.strip()
    if suspicious_filenames.duplicated().any():
        raise ValueError(f"{suspicious_labels_path} contains repeated filenames")
    unknown_suspects = set(suspicious_filenames) - set(labels["filename"])
    if unknown_suspects:
        raise ValueError(
            "suspicious-label audit references unknown filenames: "
            f"{sorted(unknown_suspects)[:10]}"
        )

    applied_suspects = set(suspicious_filenames) & set(duplicate_clean["filename"])
    cleaned = remove_suspicious_images(duplicate_clean, suspicious_filenames)

    suspected_filtered_count = len(cleaned)
    images_dir = output_dir / "images"
    sharpness_audit = preprocess_images(
        cleaned,
        source_images_dir=data_dir / "images",
        destination_images_dir=images_dir,
        minimum_sharpness=minimum_sharpness,
    )
    sharpness_audit_path = output_dir / "sharpness_audit.csv"
    sharpness_audit.to_csv(sharpness_audit_path, index=False)

    blurred_images = sharpness_audit.loc[~sharpness_audit["retained"]]
    blurred_filenames = set(blurred_images["filename"])
    cleaned = cleaned.loc[~cleaned["filename"].isin(blurred_filenames)].reset_index(
        drop=True
    )

    exclusion_rows = [
        (filename, "duplicate review", reason)
        for filename, reason in duplicate_exclusions
    ]
    exclusion_rows.extend(
        (
            filename,
            "suspected label error",
            "Flagged by the high-priority neighbor audit; removed as a candidate, "
            "not treated as a confirmed label error.",
        )
        for filename in sorted(applied_suspects)
    )
    exclusion_rows.extend(
        (
            row.filename,
            "low sharpness",
            f"Laplacian variance {row.sharpness:.6f} is below the minimum "
            f"{row.minimum_sharpness:.6f}.",
        )
        for row in blurred_images.itertuples(index=False)
    )
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        exclusion_rows, columns=["filename", "stage", "reason"]
    ).to_csv(exclusions_path, index=False)

    cleaned_labels_path = save_cleaned_dataset(
        cleaned, output_dir / "labels.csv"
    )

    train, val, test, classes_table = stratified_split(cleaned)
    split_dir = save_split_manifests(train, val, test, classes_table, split_dir)

    summary = pd.DataFrame(
        [
            {"stage": "Original manifest", "images": len(labels)},
            {"stage": "After duplicate review", "images": len(duplicate_clean)},
            {
                "stage": "After suspected-label filtering",
                "images": suspected_filtered_count,
            },
            {"stage": "After sharpness filtering", "images": len(cleaned)},
            {"stage": "Train split", "images": len(train)},
            {"stage": "Validation split", "images": len(val)},
            {"stage": "Test split", "images": len(test)},
        ]
    )
    return {
        "summary": summary,
        "cleaned_labels": cleaned,
        "train": train,
        "validation": val,
        "test": test,
        "classes": classes_table,
        "images_dir": images_dir,
        "sharpness_audit_path": sharpness_audit_path,
        "minimum_sharpness": minimum_sharpness,
        "cleaned_labels_path": cleaned_labels_path,
        "split_dir": split_dir,
        "exclusions_path": exclusions_path,
    }


if __name__ == "__main__":
    result = run_preprocessing_pipeline()
    print(result["summary"].to_string(index=False))
    print(f"Preprocessed images: {result['images_dir']}")
    print(f"Sharpness audit: {result['sharpness_audit_path']}")
    print(f"Split manifests: {result['split_dir']}")
