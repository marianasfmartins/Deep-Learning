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
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
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

    if "label" not in classes.columns or classes.empty:
        raise ValueError("classes.csv must contain a non-empty 'label' column")
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

    return train, validation, num_classes


def _make_dataset(
    frame: pd.DataFrame,
    images_dir: Path,
    preprocess_input: Callable[..., Any],
    tf: Any,
    batch_size: int,
    shuffle: bool,
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


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def compare_pretrained_cnns(
    split_dir: Path = BASE / "splits",
    images_dir: Path = BASE / "preprocessed_data" / "images",
    model_names: Iterable[str] = MODEL_NAMES,
    batch_size: int = 32,
    epochs: int = 5,
    results_path: Path | None = BASE / "splits" / "model_selection_metrics.csv",
) -> pd.DataFrame:
    """Train frozen ImageNet feature extractors and compare train/validation metrics."""
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

    train, validation, num_classes = _load_splits(split_dir, images_dir)
    tf = _require_tensorflow()
    specs = _model_specs(tf)

    tf.keras.utils.set_random_seed(SEED)
    results: list[dict[str, Any]] = []

    for model_name in selected_models:
        tf.keras.backend.clear_session()
        spec = specs[model_name]
        train_dataset = _make_dataset(
            train, images_dir, spec.preprocess_input, tf, batch_size, shuffle=True
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
                **_metric_values(frame["label"].to_numpy(), predictions),
            })

        del model, base_model

    results_frame = pd.DataFrame(results).sort_values(
        ["split", "macro_f1_score"],
        ascending=[True, False],
    ).reset_index(drop=True)
    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_frame.to_csv(results_path, index=False)
    return results_frame
