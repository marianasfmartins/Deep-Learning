"""Tune a scratch CNN on Tiny ImageNet with Keras Tuner Hyperband.

Run after preprocessing has created ``splits/`` and ``preprocessed_data/images``:
    python cnn_scratch_hyperband.py --max-epochs 9

The tuner reads only the training and validation splits. Its files are written
to a new timestamped directory under ``scratch_cnn_tuning_runs/``.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


BASE = Path(__file__).resolve().parent
IMAGE_SIZE = 64
SEED = 42


@tf.keras.utils.register_keras_serializable(package="TinyImageNet")
class MacroF1Score(tf.keras.metrics.Metric):
    """Accumulate a confusion matrix and report macro-averaged F1."""

    def __init__(self, num_classes: int, name: str = "macro_f1", **kwargs: Any):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.confusion = self.add_weight(
            name="confusion_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        labels = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
        batch_confusion = tf.math.confusion_matrix(
            labels,
            predictions,
            num_classes=self.num_classes,
            weights=weights,
            dtype=self.dtype,
        )
        self.confusion.assign_add(batch_confusion)

    def result(self) -> tf.Tensor:
        true_positives = tf.linalg.diag_part(self.confusion)
        actual = tf.reduce_sum(self.confusion, axis=1)
        predicted = tf.reduce_sum(self.confusion, axis=0)
        class_f1 = tf.math.divide_no_nan(
            2.0 * true_positives, actual + predicted
        )
        return tf.reduce_mean(class_f1)

    def reset_state(self) -> None:
        self.confusion.assign(tf.zeros_like(self.confusion))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


def _require_keras_tuner() -> Any:
    try:
        import keras_tuner as kt
    except ModuleNotFoundError as error:
        if error.name != "keras_tuner":
            raise
        raise RuntimeError(
            "Keras Tuner is required. Install the optional dependency with "
            "`python -m pip install -r requirements-cnn-tuning.txt`."
        ) from error
    return kt


def _load_splits(
    split_dir: Path, images_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train = pd.read_csv(split_dir / "train.csv")
    validation = pd.read_csv(split_dir / "val.csv")
    classes = pd.read_csv(split_dir / "classes.csv")
    required = {"filename", "label"}

    if classes.empty or "label" not in classes.columns:
        raise ValueError("classes.csv must contain a non-empty 'label' column")
    class_labels = pd.to_numeric(classes["label"], errors="raise")
    num_classes = len(classes)
    if (
        (class_labels % 1 != 0).any()
        or set(class_labels.astype(int)) != set(range(num_classes))
    ):
        raise ValueError(
            "classes.csv labels must be unique integers from 0 to num_classes - 1"
        )

    expected_labels = set(range(num_classes))
    for split_name, frame in (("train", train), ("validation", validation)):
        missing_columns = required - set(frame.columns)
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
        if set(frame["label"].unique()) != expected_labels:
            raise ValueError(
                f"{split_name} split must contain every class label from "
                f"0 through {num_classes - 1}"
            )

    overlap = set(train["filename"]) & set(validation["filename"])
    if overlap:
        raise ValueError(
            f"Training and validation splits overlap: {sorted(overlap)[:5]}"
        )

    filenames = pd.concat(
        [train["filename"], validation["filename"]], ignore_index=True
    )
    invalid = [name for name in filenames if Path(name).name != name]
    if invalid:
        raise ValueError(
            f"Split manifests must contain filenames, not paths: {invalid[:5]}"
        )
    missing = [
        name for name in filenames.unique() if not (images_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} split images are missing from {images_dir}; "
            f"examples: {missing[:5]}"
        )
    return train, validation, num_classes


def _make_dataset(
    frame: pd.DataFrame,
    images_dir: Path,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    paths = [str(images_dir / name) for name in frame["filename"]]
    labels = frame["label"].to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(frame), 8192),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def decode_jpeg(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    return (
        dataset.map(decode_jpeg, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _add_activation(x: tf.Tensor, activation: str, name: str) -> tf.Tensor:
    if activation == "leaky_relu":
        return tf.keras.layers.LeakyReLU(negative_slope=0.1, name=name)(x)
    return tf.keras.layers.Activation(activation, name=name)(x)


def _build_model(hp: Any, num_classes: int) -> tf.keras.Model:
    num_blocks = hp.Int("conv_blocks", min_value=2, max_value=4)
    base_filters = hp.Choice("base_filters", values=[32, 64])
    kernel_size = hp.Choice("kernel_size", values=[3, 5])
    activation = hp.Choice(
        "activation", values=["relu", "elu", "selu", "leaky_relu"]
    )
    pooling = hp.Choice(
        "pooling",
        values=["max_pooling", "average_pooling", "global_average_pooling"],
    )
    dense_layers = hp.Int("dense_layers", min_value=1, max_value=3)
    dense_units = hp.Choice("dense_units", values=[64, 128, 256, 512])
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.0, max_value=0.5, step=0.1
    )
    use_batch_norm = hp.Boolean("batch_normalization")
    l1 = hp.Float("l1", min_value=1e-5, max_value=1e-2, sampling="log")
    l2 = hp.Float("l2", min_value=1e-5, max_value=1e-2, sampling="log")
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )
    optimizer_name = hp.Choice("optimizer", values=["adam", "sgd", "rmsprop"])

    regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    x = inputs
    for block in range(num_blocks):
        filters = min(base_filters * (2**block), 256)
        for conv in range(2):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                kernel_regularizer=regularizer,
                name=f"block_{block + 1}_conv_{conv + 1}",
            )(x)
            if use_batch_norm:
                x = tf.keras.layers.BatchNormalization(
                    name=f"block_{block + 1}_bn_{conv + 1}"
                )(x)
            x = _add_activation(
                x, activation, name=f"block_{block + 1}_activation_{conv + 1}"
            )
        if pooling == "average_pooling":
            x = tf.keras.layers.AveragePooling2D(
                pool_size=2, name=f"block_{block + 1}_average_pool"
            )(x)
        else:
            x = tf.keras.layers.MaxPooling2D(
                pool_size=2, name=f"block_{block + 1}_max_pool"
            )(x)
        if dropout_rate:
            x = tf.keras.layers.Dropout(
                dropout_rate, name=f"block_{block + 1}_dropout"
            )(x)

    if pooling == "global_average_pooling":
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    else:
        x = tf.keras.layers.Flatten(name="flatten")(x)

    for dense_index in range(dense_layers):
        x = tf.keras.layers.Dense(
            dense_units,
            kernel_regularizer=regularizer,
            name=f"dense_{dense_index + 1}",
        )(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(
                name=f"dense_{dense_index + 1}_bn"
            )(x)
        x = _add_activation(
            x, activation, name=f"dense_{dense_index + 1}_activation"
        )
        if dropout_rate:
            x = tf.keras.layers.Dropout(
                dropout_rate, name=f"dense_{dense_index + 1}_dropout"
            )(x)

    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tiny_imagenet_cnn")

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9
        )
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            MacroF1Score(num_classes=num_classes),
        ],
    )
    return model


def run_hyperband(
    split_dir: Path = BASE / "splits",
    images_dir: Path = BASE / "preprocessed_data" / "images",
    output_dir: Path | None = None,
    max_epochs: int = 9,
    overwrite: bool = False,
) -> dict[str, Any]:
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    kt = _require_keras_tuner()
    split_dir = Path(split_dir)
    images_dir = Path(images_dir)
    if output_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = BASE / "scratch_cnn_tuning_runs" / run_id
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose another path "
            "or pass overwrite=True to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train, validation, num_classes = _load_splits(split_dir, images_dir)
    tf.keras.utils.set_random_seed(SEED)

    class ScratchCNNHyperModel(kt.HyperModel):
        def build(self, hp: Any) -> tf.keras.Model:
            return _build_model(hp, num_classes)

        def fit(self, hp: Any, model: tf.keras.Model, *args: Any, **kwargs: Any):
            batch_size = hp.Choice("batch_size", values=[16, 32, 64])
            train_dataset = _make_dataset(
                train, images_dir, batch_size=batch_size, shuffle=True
            )
            validation_dataset = _make_dataset(
                validation, images_dir, batch_size=batch_size, shuffle=False
            )
            return model.fit(
                train_dataset,
                validation_data=validation_dataset,
                *args,
                **kwargs,
            )

    tuner = kt.Hyperband(
        hypermodel=ScratchCNNHyperModel(),
        objective=kt.Objective("val_macro_f1", direction="max"),
        max_epochs=max_epochs,
        factor=3,
        hyperband_iterations=1,
        directory=str(output_dir),
        project_name="hyperband",
        overwrite=overwrite,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    tuner.search(epochs=max_epochs, callbacks=callbacks, verbose=2)

    best_trials = tuner.oracle.get_best_trials(num_trials=1)
    if not best_trials or best_trials[0].score is None:
        raise RuntimeError(
            "Hyperband completed without a scored trial; inspect the tuner "
            f"records in {output_dir} for trial failures."
        )
    best_trial = best_trials[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    model_path = output_dir / "best_scratch_cnn.keras"
    best_model.save(model_path)

    best_batch_size = int(best_trial.hyperparameters.get("batch_size"))
    validation_dataset = _make_dataset(
        validation, images_dir, batch_size=best_batch_size, shuffle=False
    )
    validation_metrics = best_model.evaluate(
        validation_dataset, verbose=0, return_dict=True
    )
    summary = {
        "objective": "val_macro_f1",
        "best_trial_id": best_trial.trial_id,
        "best_trial_score": float(best_trial.score),
        "best_hyperparameters": best_trial.hyperparameters.values,
        "validation_metrics": {
            name: float(value) for name, value in validation_metrics.items()
        },
        "best_model_path": str(model_path),
    }
    summary_path = output_dir / "best_trial_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    trials = []
    for trial in tuner.oracle.trials.values():
        trials.append({
            "trial_id": trial.trial_id,
            "status": trial.status,
            "score": trial.score,
            **trial.hyperparameters.values,
        })
    pd.DataFrame(trials).to_csv(output_dir / "trials.csv", index=False)
    tuner.results_summary()
    return {
        "summary": summary,
        "summary_path": summary_path,
        "trials_path": output_dir / "trials.csv",
        "model_path": model_path,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=9,
        help="maximum Hyperband epochs per trial (default: 9)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory; otherwise a unique timestamped directory is used",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reuse/overwrite tuner artifacts in the selected output directory",
    )
    args = parser.parse_args()
    result = run_hyperband(
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        overwrite=args.overwrite,
    )
    print(f"Best trial summary: {result['summary_path']}")
    print(f"Best model: {result['model_path']}")
    print(f"Trial results: {result['trials_path']}")


if __name__ == "__main__":
    main()
