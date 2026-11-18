"""Model-selection utilities for Tiny ImageNet transfer-learning experiments."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


BASE = Path(__file__).resolve().parent
IMAGE_SIZE = 224
SEED = 42
MODEL_NAMES = ("VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0")


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., Any]
    preprocess_input: Callable[..., Any]


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        if error.name != "tensorflow":
            raise
        raise RuntimeError(
            "TensorFlow is required for model selection. Install it in the "
            "notebook's active Python environment before running this experiment."
        ) from error
    return tf


def _model_specs(tf: Any) -> dict[str, ModelSpec]:
    applications = tf.keras.applications
    return {
        "VGG16": ModelSpec(
            applications.VGG16,
            applications.vgg16.preprocess_input,
        ),
        "ResNet50": ModelSpec(
            applications.ResNet50,
            applications.resnet50.preprocess_input,
        ),
        "MobileNetV2": ModelSpec(
            applications.MobileNetV2,
            applications.mobilenet_v2.preprocess_input,
        ),
        "EfficientNetB0": ModelSpec(
            applications.EfficientNetB0,
            applications.efficientnet.preprocess_input,
        ),
    }


def _load_splits(
    split_dir: Path, images_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    required_split_columns = {"filename", "label"}
    train = pd.read_csv(split_dir / "train.csv")
    validation = pd.read_csv(split_dir / "val.csv")
    classes = pd.read_csv(split_dir / "classes.csv")

    for split_name, frame in (("train", train), ("validation", validation)):
        missing_columns = required_split_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{split_name}.csv is missing columns: {sorted(missing_columns)}"
            )
        if frame.empty:
            raise ValueError(f"{split_name}.csv must contain at least one image")
        if frame["filename"].duplicated().any():
            raise ValueError(f"{split_name}.csv contains duplicate filenames")

        labels = pd.to_numeric(frame["label"], errors="raise")
        if (labels % 1 != 0).any():
            raise ValueError(f"{split_name}.csv contains non-integer labels")
        frame["filename"] = frame["filename"].astype(str)
        frame["label"] = labels.astype(np.int32)

    if classes.empty:
        raise ValueError("classes.csv must not be empty")
    missing_class_columns = {"label", "class_id", "description"} - set(classes.columns)
    if missing_class_columns:
        raise ValueError(
            f"classes.csv is missing columns: {sorted(missing_class_columns)}"
        )
    num_classes = len(classes)
    expected_labels = set(range(num_classes))
    class_labels = pd.to_numeric(classes["label"], errors="raise")
    if (
        (class_labels % 1 != 0).any()
        or set(class_labels.astype(int)) != expected_labels
    ):
        raise ValueError(
            "classes.csv labels must be unique integers from 0 to num_classes - 1"
        )
    classes = classes.copy()
    classes["label"] = class_labels.astype(np.int32)
    classes = classes.sort_values("label").reset_index(drop=True)
    for split_name, frame in (("train", train), ("validation", validation)):
        actual_labels = set(frame["label"].unique())
        if actual_labels != expected_labels:
            raise ValueError(
                f"{split_name} split must contain labels 0 through {num_classes - 1}"
            )

    overlapping_files = set(train["filename"]) & set(validation["filename"])
    if overlapping_files:
        raise ValueError(
            "Training and validation splits overlap; examples: "
            f"{sorted(overlapping_files)[:5]}"
        )

    filenames = pd.concat([train["filename"], validation["filename"]], ignore_index=True)
    invalid_filenames = [name for name in filenames if Path(name).name != name]
    if invalid_filenames:
        raise ValueError(
            "Split manifests must contain filenames, not paths; examples: "
            f"{invalid_filenames[:5]}"
        )

    missing_files = [
        name for name in filenames.unique()
        if not (images_dir / name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} split images are missing from {images_dir}; "
            f"examples: {missing_files[:5]}"
        )

    return train, validation, num_classes, classes


def _oversample_training_data(
    train: pd.DataFrame,
    min_samples_per_class: int = 200,
) -> pd.DataFrame:
    """Randomly duplicate minority-class rows up to a training-count floor."""
    if min_samples_per_class <= 0:
        raise ValueError("min_samples_per_class must be positive")
    if "label" not in train.columns:
        raise ValueError("training data is missing the required 'label' column")
    if train.empty:
        raise ValueError("cannot oversample an empty training dataset")

    counts = train["label"].value_counts()
    rng = np.random.default_rng(SEED)
    extra_rows = []
    for label, count in counts.items():
        if count < min_samples_per_class:
            class_rows = train.loc[train["label"] == label]
            sampled_indices = rng.choice(
                class_rows.index.to_numpy(),
                size=min_samples_per_class - int(count),
                replace=True,
            )
            extra_rows.append(train.loc[sampled_indices])

    if not extra_rows:
        return train.copy().reset_index(drop=True)
    return pd.concat([train, *extra_rows], ignore_index=True).sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)


def _build_training_augmentation() -> Any:
    """Create modest online augmentations suited to small natural images."""
    try:
        import albumentations as A
    except ModuleNotFoundError as error:
        if error.name != "albumentations":
            raise
        raise RuntimeError(
            "Albumentations is required for online training augmentation. "
            "Install the optional dependencies with "
            "`python -m pip install -r requirements-modeling.txt`."
        ) from error

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3,
            ),
        ],
        seed=SEED,
    )


def _augment_numpy_image(image: Any, augmentation: Any) -> np.ndarray:
    """Apply one fresh Albumentations transform to a decoded uint8 image."""
    image_array = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    augmented = augmentation(image=image_array)["image"]
    return np.asarray(augmented, dtype=np.uint8)


def _make_dataset(
    frame: pd.DataFrame,
    images_dir: Path,
    preprocess_input: Callable[..., Any],
    tf: Any,
    batch_size: int,
    shuffle: bool,
    augmentation: Any | None = None,
) -> Any:
    filenames = frame["filename"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)
    paths = np.asarray([str(images_dir / name) for name in filenames])
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(frame), 8192),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def decode_and_resize(path: Any, label: Any) -> tuple[Any, Any]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        if augmentation is not None:
            image = tf.py_function(
                func=lambda decoded: _augment_numpy_image(decoded, augmentation),
                inp=[image],
                Tout=tf.uint8,
            )
            image.set_shape([None, None, 3])
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        return tf.cast(image, tf.float32), label

    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)

    def preprocess_batch(images: Any, batch_labels: Any) -> tuple[Any, Any]:
        return preprocess_input(images), batch_labels

    return dataset.map(
        preprocess_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_f1_score": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
    }


def compare_pretrained_cnns(
    split_dir: Path = BASE / "splits",
    images_dir: Path = BASE / "preprocessed_data" / "images",
    model_names: Iterable[str] = MODEL_NAMES,
    batch_size: int = 32,
    epochs: int = 5,
    results_path: Path | None = BASE / "splits" / "model_selection_metrics.csv",
    per_class_results_path: Path | None = (
        BASE / "splits" / "model_selection_per_class_recall.csv"
    ),
    oversample_minority_classes: bool = True,
    min_samples_per_class: int = 200,
    augment_training: bool = True,
) -> pd.DataFrame:
    """Train pretrained CNNs and compare train/validation classification metrics.

    Minority training classes are randomly oversampled to a sample-count floor,
    with online Albumentations applied after JPEG decoding. Validation and the
    train-set metric pass retain their original rows and are never augmented.
    """
    split_dir = Path(split_dir)
    images_dir = Path(images_dir)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if isinstance(model_names, str):
        raise TypeError("model_names must be a sequence of names, not one string")
    selected_models = tuple(model_names)
    if not selected_models:
        raise ValueError("select at least one model")
    if len(set(selected_models)) != len(selected_models):
        raise ValueError("model_names must not contain duplicate model names")
    unknown_models = set(selected_models) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"unknown model names: {sorted(unknown_models)}")

    train, validation, num_classes, classes = _load_splits(split_dir, images_dir)
    training_rows = (
        _oversample_training_data(train, min_samples_per_class)
        if oversample_minority_classes
        else train.copy()
    )
    if oversample_minority_classes:
        print(
            f"Training rows: {len(train):,} -> {len(training_rows):,}; "
            f"classes below {min_samples_per_class} samples were oversampled."
        )
    augmentation = _build_training_augmentation() if augment_training else None
    tf = _require_tensorflow()
    specs = _model_specs(tf)

    tf.keras.utils.set_random_seed(SEED)
    results: list[dict[str, Any]] = []
    per_class_results: list[dict[str, Any]] = []
    class_metadata = classes.set_index("label")

    for model_name in selected_models:
        tf.keras.backend.clear_session()
        spec = specs[model_name]
        train_dataset = _make_dataset(
            training_rows,
            images_dir,
            spec.preprocess_input,
            tf,
            batch_size,
            shuffle=True,
            augmentation=augmentation,
        )
        train_evaluation_dataset = _make_dataset(
            train, images_dir, spec.preprocess_input, tf, batch_size, shuffle=False
        )
        validation_dataset = _make_dataset(
            validation, images_dir, spec.preprocess_input, tf, batch_size, shuffle=False
        )

        base_model = spec.builder(
            include_top=False,
            weights="imagenet",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            pooling="avg",
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        features = base_model(inputs, training=False)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                )
            ],
            verbose=2,
        )

        for split_name, dataset, frame in (
            ("train", train_evaluation_dataset, train),
            ("validation", validation_dataset, validation),
        ):
            probabilities = model.predict(dataset, verbose=0)
            predictions = np.argmax(probabilities, axis=1)
            if len(predictions) != len(frame):
                raise RuntimeError(
                    f"{model_name} produced {len(predictions)} predictions for "
                    f"{len(frame)} {split_name} samples"
                )
            results.append({
                "model": model_name,
                "split": split_name,
                **_metric_values(
                    frame["label"].to_numpy(), predictions, num_classes
                ),
            })
            class_recalls = recall_score(
                frame["label"].to_numpy(),
                predictions,
                labels=np.arange(num_classes),
                average=None,
                zero_division=0,
            )
            per_class_results.extend(
                {
                    "model": model_name,
                    "split": split_name,
                    "label": label,
                    "class_id": class_metadata.loc[label, "class_id"],
                    "description": class_metadata.loc[label, "description"],
                    "recall": float(class_recalls[label]),
                }
                for label in range(num_classes)
            )

        del model, base_model

    results_frame = pd.DataFrame(results).sort_values(
        ["split", "macro_f1_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_frame.to_csv(results_path, index=False)
    if per_class_results_path is not None:
        per_class_results_path = Path(per_class_results_path)
        per_class_results_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_class_results).to_csv(per_class_results_path, index=False)
    return results_frame
