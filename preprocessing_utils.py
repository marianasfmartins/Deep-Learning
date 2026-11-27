"""Reusable preprocessing helpers for the Tiny ImageNet notebook."""

import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.restoration import richardson_lucy
from sklearn.model_selection import StratifiedShuffleSplit


BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
IMAGES_DIR = DATA_DIR / "images"
RGB_IMAGES_DIR = BASE / "preprocessed_data" / "rgb"         # all images as RGB (before the split)
PREPROCESSED_IMAGES_DIR = BASE / "preprocessed_data" / "images"  # final, restored images
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


def _require_columns(frame, required, name):
    missing_columns = set(required) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")


def remove_duplicates(labels, duplicate_decisions):
    """Return a copy of ``labels`` without the images marked ``exclude`` in the duplicate review.

    ``duplicate_decisions`` is the reviewed table (``duplicate_resolution.csv``) with
    one row per duplicate candidate and a ``decision`` of ``keep`` or ``exclude``.
    """
    _require_columns(labels, {"filename"}, "labels")
    _require_columns(duplicate_decisions, {"filename", "decision"}, "duplicate decisions")

    decisions = duplicate_decisions.copy()
    decisions["filename"] = decisions["filename"].astype(str).str.strip()
    decisions["decision"] = decisions["decision"].fillna("").astype(str).str.strip().str.lower()
    invalid = set(decisions["decision"]) - {"keep", "exclude"}
    if invalid:
        raise ValueError(f"duplicate decisions must be 'keep' or 'exclude'; found {sorted(invalid)}")
    unknown = set(decisions["filename"]) - set(labels["filename"].astype(str))
    if unknown:
        raise ValueError(f"duplicate decisions reference unknown filenames: {sorted(unknown)[:10]}")

    excluded = set(decisions.loc[decisions["decision"] == "exclude", "filename"])
    cleaned = labels.loc[~labels["filename"].astype(str).isin(excluded)].copy()
    return cleaned.reset_index(drop=True)


def convert_to_rgb(labels, output_dir=RGB_IMAGES_DIR, source_dir=IMAGES_DIR, n_workers=8):
    """Write every image of ``labels`` to ``output_dir`` as a 3-channel RGB image.

    RGB files are copied byte for byte (no re-compression). Files in any other
    mode (grayscale ``L`` in this dataset) are converted, which copies the gray
    value into all three channels, and saved as JPEG quality 95. Returns a copy
    of ``labels`` with the image's ``original_mode``.
    """
    _require_columns(labels, {"filename"}, "labels")
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("source and output directories must be different")
    output_dir.mkdir(parents=True, exist_ok=True)

    def process(filename):
        source = source_dir / filename
        with Image.open(source) as image:
            mode = image.mode
        if mode == "RGB":
            shutil.copyfile(source, output_dir / filename)
        else:
            write_rgb(output_dir / filename, read_rgb(source))
        return mode

    filenames = labels["filename"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        modes = list(pool.map(process, filenames))
    converted = labels.copy()
    converted["original_mode"] = modes
    return converted


def load_embeddings(path, filenames):
    """Return the saved embeddings of ``filenames``, in that order (rows of a float array)."""
    saved = np.load(path, allow_pickle=False)
    index = {str(name): row for row, name in enumerate(saved["filenames"])}
    missing = [name for name in filenames if str(name) not in index]
    if missing:
        raise ValueError(f"{len(missing)} images have no saved embedding, e.g. {missing[:5]}")
    return saved["embeddings"][[index[str(name)] for name in filenames]]


def prepare_minority_training_rows(
    training_rows,
    y_train,
    min_samples_per_class=200,
    seed=SEED,
):
    """Create marked training-only copies for classes below a count floor.

    The returned rows identify copies with ``is_augmented`` so the modeling
    pipeline can apply Albumentations only to those samples.
    """
    required = {"filename", "label"}
    missing_columns = required - set(training_rows.columns)
    if missing_columns:
        raise ValueError(
            f"training data is missing columns: {sorted(missing_columns)}"
        )
    if (
        isinstance(min_samples_per_class, bool)
        or not isinstance(min_samples_per_class, (int, np.integer))
        or min_samples_per_class <= 0
    ):
        raise ValueError("min_samples_per_class must be a positive integer")

    y_train = np.asarray(y_train)
    if y_train.ndim != 1 or len(y_train) != len(training_rows):
        raise ValueError("y_train must be one-dimensional and match training_rows")
    if len(y_train) == 0:
        raise ValueError("cannot augment an empty training dataset")
    if pd.isna(y_train).any():
        raise ValueError("y_train must not contain missing labels")

    row_labels = pd.to_numeric(training_rows["label"], errors="raise").to_numpy()
    if not np.array_equal(row_labels, y_train):
        raise ValueError("y_train must match the labels in training_rows")

    prepared_rows = training_rows.copy()
    prepared_rows["is_augmented"] = False
    rng = np.random.default_rng(seed)
    additional_rows = []
    class_counts = pd.Series(y_train).value_counts()
    for label, count in class_counts.items():
        if count >= min_samples_per_class:
            continue
        class_positions = np.flatnonzero(y_train == label)
        sampled_positions = rng.choice(
            class_positions,
            size=min_samples_per_class - int(count),
            replace=True,
        )
        class_copies = training_rows.iloc[sampled_positions].copy()
        class_copies["is_augmented"] = True
        additional_rows.append(class_copies)

    if not additional_rows:
        return prepared_rows.reset_index(drop=True)
    return pd.concat([prepared_rows, *additional_rows], ignore_index=True).sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)


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


# =============================================================================
# Image quality: reading, measuring, thresholds, fixes and calibration helpers
# =============================================================================

def read_rgb(path):
    """Read an image as an RGB uint8 array (grayscale files come back with 3 equal channels)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb, jpeg_quality=95):
    """Save an RGB uint8 array (OpenCV writes BGR, so the channels are swapped first)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
        raise OSError(f"cannot write image: {path}")


def _immerkaer_noise_sigma(channel):
    """Noise standard deviation of one channel (Immerkaer, 1996).

    The 3x3 kernel cancels smooth gradients and straight edges, so what is left
    is mostly pixel-to-pixel noise.
    """
    c = channel
    response = (
        c[:-2, :-2] - 2 * c[:-2, 1:-1] + c[:-2, 2:]
        - 2 * c[1:-1, :-2] + 4 * c[1:-1, 1:-1] - 2 * c[1:-1, 2:]
        + c[2:, :-2] - 2 * c[2:, 1:-1] + c[2:, 2:]
    )
    height, width = c.shape
    return float(np.sqrt(np.pi / 2) * np.abs(response).sum() / (6 * (width - 2) * (height - 2)))


def measure_color_noise(rgb):
    """Estimate per-pixel colour noise of an RGB array (H x W x 3).

    Real texture changes all three channels together, so it mostly cancels in
    the colour differences R-G and B-G. Sensor-like or added noise is
    independent per channel and survives there. Measuring noise on R-G and B-G
    therefore separates noise from fine texture. Grayscale images score 0.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (_immerkaer_noise_sigma(red - green) + _immerkaer_noise_sigma(blue - green)) / 2


def _edge_step_profile(gray):
    """Mean absolute step between pixel p and p+1, for each position p (both axes pooled)."""
    horizontal = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    vertical = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    return (horizontal + vertical) / 2


def measure_jpeg_blockiness(gray, eps=0.5):
    """How much stronger edges are on the 8x8 JPEG block grid than inside blocks.

    Heavy JPEG compression makes each 8x8 block flat on its own, so steps appear
    at the block borders. Values near 1 mean no visible blocks; clearly above 1
    means block artifacts. Comparing with the strongest inside position keeps
    pixelated images (whose steps also fall inside 8x8 blocks) from scoring here.
    """
    profile = _edge_step_profile(np.asarray(gray, dtype=np.float32))
    position = np.arange(len(profile)) % 8
    border = profile[position == 7].mean()
    strongest_inside = max(profile[position == offset].mean() for offset in range(7))
    return float((border + eps) / (strongest_inside + eps))


def measure_pixelation(gray, factors=range(2, 8), eps=0.5):
    """Detect nearest-neighbour upscaling (images made of f x f pixel blocks).

    For each block size f, compare the typical step at block borders with the
    typical step inside blocks; pixelated images have steps only at borders.
    Medians over positions ignore single strong edges such as frame borders.
    Returns the largest ratio over f (about 1 = not pixelated).
    """
    profile = _edge_step_profile(np.asarray(gray, dtype=np.float32))
    index = np.arange(len(profile))
    best = 1.0
    for factor in factors:
        border = (index + 1) % factor == 0
        ratio = (np.median(profile[border]) + eps) / (np.median(profile[~border]) + eps)
        best = max(best, float(ratio))
    return best


QUALITY_MEASURES = ["brightness", "contrast", "sharpness", "noise", "blockiness", "pixelation"]


def measure_quality(rgb):
    """The six image-quality measures of one RGB uint8 image.

    brightness / contrast: mean and standard deviation of the L channel (LAB);
    sharpness: variance of the Laplacian; noise, blockiness, pixelation: as in the EDA.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    return {
        "brightness": float(lightness.mean()),
        "contrast": float(lightness.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise": measure_color_noise(rgb),
        "blockiness": measure_jpeg_blockiness(gray),
        "pixelation": measure_pixelation(gray),
    }


def iqr_cutoffs(values, log_scale=False, k=1.5):
    """Return (low, high) outlier cutoffs using the boxplot (Tukey IQR) rule.

    A value is an outlier when it lies more than ``k`` x IQR below the 25th
    percentile (Q1) or above the 75th percentile (Q3). With ``log_scale`` the
    rule is applied to log10 of the values and the cutoffs are converted back,
    which suits measures that span several orders of magnitude (sharpness).
    """
    values = np.asarray(pd.Series(values, dtype=float).dropna())
    if log_scale:
        values = np.log10(np.clip(values, 1e-6, None))
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    if log_scale:
        low, high = 10 ** low, 10 ** high
    return float(low), float(high)


# ---- Fixes -------------------------------------------------------------------

def gaussian_psf(sigma):
    """Normalised 2-D Gaussian point-spread function (the blur kernel assumed by deconvolution)."""
    size = 2 * int(np.ceil(3 * sigma)) + 1
    kernel = cv2.getGaussianKernel(size, sigma)
    psf = kernel @ kernel.T
    return psf / psf.sum()


def deblur_image(rgb, psf_sigma, iterations):
    """Richardson-Lucy deconvolution with a Gaussian PSF, per channel.

    The image is padded by reflection first, so the deconvolution does not
    create ringing at the borders, and cropped back afterwards.
    """
    psf = gaussian_psf(psf_sigma)
    pad = psf.shape[0]
    padded = np.pad(rgb.astype(np.float64) / 255.0, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    restored = np.stack(
        [richardson_lucy(padded[..., channel], psf, num_iter=int(iterations), clip=True)
         for channel in range(3)],
        axis=-1,
    )[pad:-pad, pad:-pad]
    return np.clip(restored * 255.0 + 0.5, 0, 255).astype(np.uint8)


def denoise_image(rgb, strength):
    """Non-local-means colour denoising; ``strength`` is OpenCV's filter strength h."""
    return cv2.fastNlMeansDenoisingColored(rgb, None, float(strength), float(strength), 7, 21)


def adjust_gamma(rgb, current_brightness, target_brightness, gamma_range=(0.4, 2.5)):
    """Gamma curve (via a lookup table) that moves the brightness towards ``target_brightness``.

    Brightness values are on the 0-255 L scale. gamma < 1 brightens, gamma > 1 darkens.
    """
    current = np.clip(current_brightness / 255.0, 0.01, 0.99)
    target = np.clip(target_brightness / 255.0, 0.01, 0.99)
    gamma = float(np.clip(np.log(target) / np.log(current), *gamma_range))
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(rgb, table)


def enhance_contrast(rgb, clip_limit=2.0, tile_grid=(8, 8)):
    """CLAHE on the L channel of LAB, so colours are not shifted."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def compress_contrast(rgb, current_contrast, target_contrast):
    """Scale the L channel around its mean so its standard deviation becomes ``target_contrast``."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]
    alpha = float(np.clip(target_contrast / max(current_contrast, 1e-6), 0.0, 1.0))
    beta = float(lightness.mean()) * (1.0 - alpha)
    lab[..., 0] = cv2.convertScaleAbs(lightness, alpha=alpha, beta=beta)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def deblock_image(rgb, diameter=5, sigma_color=25, sigma_space=5):
    """Edge-preserving bilateral filter that smooths JPEG block borders."""
    return cv2.bilateralFilter(rgb, int(diameter), float(sigma_color), float(sigma_space))


def depixelate_image(rgb, kernel_size=3):
    """Median filter that rounds off the hard edges of enlarged pixel blocks."""
    return cv2.medianBlur(rgb, int(kernel_size))


# ---- Synthetic degradations and similarity (calibration) ----------------------

def add_gaussian_blur(rgb, sigma):
    return cv2.GaussianBlur(rgb, (0, 0), float(sigma))


def add_gaussian_noise(rgb, sigma, rng):
    noisy = rgb.astype(np.float32) + rng.normal(0.0, float(sigma), rgb.shape)
    return np.clip(noisy + 0.5, 0, 255).astype(np.uint8)


def psnr(reference, image):
    """Peak signal-to-noise ratio in dB (higher = closer to the reference)."""
    return float(cv2.PSNR(reference, image))


