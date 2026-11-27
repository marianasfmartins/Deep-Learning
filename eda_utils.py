"""Reusable image-audit helpers for the Tiny ImageNet notebook."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from preprocessing_utils import measure_image_sharpness


BASE = Path(__file__).resolve().parent
IMAGES_DIR = BASE / "data" / "images"
PHASH_THRESHOLD = 4


def inspect_image(filename, images_dir=IMAGES_DIR):
    """Fully decode one image and report what was found.

    ``Image.open`` only reads the file header, so a truncated or half-written
    JPEG would still look fine. ``verify`` checks the file structure and
    ``load`` decodes every pixel, which is what catches real corruption.
    """
    path = Path(images_dir) / str(filename)
    result = {
        "filename": str(filename),
        "exists": path.is_file(),
        "file_bytes": 0,
        "file_format": None,
        "width": None,
        "height": None,
        "color_mode": None,
        "readable": False,
        "error": "",
    }
    if not result["exists"]:
        result["error"] = "file not found"
        return result

    result["file_bytes"] = path.stat().st_size
    if result["file_bytes"] == 0:
        result["error"] = "empty file (0 bytes)"
        return result

    try:
        with Image.open(path) as image:
            image.verify()
        # verify() leaves the image unusable, so open it again to decode the pixels.
        with Image.open(path) as image:
            image.load()
            result.update(
                file_format=image.format,
                width=image.width,
                height=image.height,
                color_mode=image.mode,
                readable=True,
            )
    except (OSError, ValueError, SyntaxError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def get_hash(filename, images_dir=IMAGES_DIR):
    return hashlib.md5((Path(images_dir) / str(filename)).read_bytes()).hexdigest()


def compute_perceptual_hash(image_path):
    with Image.open(image_path) as image:
        pixels = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    coefficients = dctn(pixels, type=2, norm="ortho")
    low_frequency = coefficients[:8, :8]
    median = np.median(low_frequency.reshape(-1)[1:])
    bits = (low_frequency >= median).reshape(-1)
    return int.from_bytes(np.packbits(bits).tobytes(), byteorder="big")


def _hamming_distance(hash_a, hash_b):
    return bin(int(hash_a) ^ int(hash_b)).count("1")


def find_perceptual_duplicate_pairs(phashes, threshold=PHASH_THRESHOLD):
    if not phashes:
        return []

    children = [{}]
    pairs = []

    for current_index in range(1, len(phashes)):
        current_hash = phashes[current_index]
        pending_nodes = [0]

        while pending_nodes:
            node_index = pending_nodes.pop()
            distance = _hamming_distance(current_hash, phashes[node_index])
            if distance <= threshold:
                pairs.append((node_index, current_index, distance))

            lower_bound = distance - threshold
            upper_bound = distance + threshold
            pending_nodes.extend(
                child_index
                for edge_distance, child_index in children[node_index].items()
                if lower_bound <= edge_distance <= upper_bound
            )

        node_index = 0
        while True:
            distance = _hamming_distance(current_hash, phashes[node_index])
            child_index = children[node_index].get(distance)
            if child_index is None:
                children[node_index][distance] = current_index
                children.append({})
                break
            node_index = child_index

    return pairs


# Standard JPEG luminance quantization table (quality 50, ITU-T T.81 Annex K).
_STANDARD_LUMA_TABLE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)

IMAGE_STAT_COLUMNS = [
    "brightness", "contrast", "sharpness", "noise", "blockiness", "pixelation", "jpeg_quality",
]


def estimate_jpeg_quality(image):
    """Estimate the quality (1-100) a JPEG was last saved with, from its file header.

    Encoders such as libjpeg scale one standard luminance table by the chosen
    quality. Comparing the file's table with the standard one inverts that
    scaling. Returns NaN for files that are not JPEGs.
    """
    tables = getattr(image, "quantization", None)
    if not tables or 0 not in tables:
        return float("nan")
    scale = 100.0 * np.sum(tables[0]) / _STANDARD_LUMA_TABLE.sum()
    quality = 5000.0 / scale if scale > 100 else (200.0 - scale) / 2.0
    return float(np.clip(quality, 1, 100))


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
    """Estimate per-pixel colour noise of an RGB array (H x W x 3, float).

    Real texture changes all three channels together, so it mostly cancels in
    the colour differences R-G and B-G. Sensor-like or added noise is
    independent per channel and survives there. Measuring noise on R-G and B-G
    therefore separates noise from fine texture. Grayscale images score 0.
    """
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
    profile = _edge_step_profile(gray)
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
    profile = _edge_step_profile(gray)
    index = np.arange(len(profile))
    best = 1.0
    for factor in factors:
        border = (index + 1) % factor == 0
        ratio = (np.median(profile[border]) + eps) / (np.median(profile[~border]) + eps)
        best = max(best, float(ratio))
    return best


def get_image_stats(filename, images_dir=IMAGES_DIR):
    """Return the image-quality metrics of one image as a dict (values None if unreadable)."""
    try:
        with Image.open(Path(images_dir) / str(filename)) as image:
            jpeg_quality = estimate_jpeg_quality(image)
            sharpness = measure_image_sharpness(image)
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            gray = np.asarray(image.convert("L"), dtype=np.float32)
    except (OSError, ValueError):
        return dict.fromkeys(IMAGE_STAT_COLUMNS)
    return {
        "brightness": float(rgb.mean()),
        "contrast": float(rgb.std()),
        "sharpness": sharpness,
        "noise": measure_color_noise(rgb),
        "blockiness": measure_jpeg_blockiness(gray),
        "pixelation": measure_pixelation(gray),
        "jpeg_quality": jpeg_quality,
    }


def iqr_cutoffs(values, log_scale=False, k=1.5):
    """Return (low, high) outlier cutoffs using the boxplot (Tukey IQR) rule.

    A value is an outlier when it lies more than ``k`` x IQR below the 25th
    percentile (Q1) or above the 75th percentile (Q3). With ``log_scale`` the
    rule is applied to log10 of the values and the cutoffs are converted back,
    which suits metrics that span several orders of magnitude (sharpness).
    """
    values = pd.Series(values, dtype=float).dropna()
    if log_scale:
        values = np.log10(values.clip(lower=1e-6))
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    if log_scale:
        low, high = 10 ** low, 10 ** high
    return float(low), float(high)


def _format_cutoff(value):
    return f"{value:,.2f}" if abs(value) < 10 else f"{value:,.1f}"


def plot_quality_distributions(
    dataframe,
    cutoffs,
    log_scale_metrics=("sharpness", "noise", "pixelation"),
    n_cols=3,
    log_floor=0.01,
):
    """Histogram each quality metric, with its cutoffs and flagged areas shaded.

    On log-scaled panels, values below ``log_floor`` (e.g. a noise of 0 for
    grayscale images) are drawn at ``log_floor``.
    """
    metrics = list(cutoffs)
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    for ax, metric in zip(axes.flat, metrics):
        raw_values = dataframe[metric].dropna()
        low, high = cutoffs[metric]
        n_low, n_high = int((raw_values < low).sum()), int((raw_values > high).sum())
        if metric in log_scale_metrics:
            values = raw_values.clip(lower=log_floor)
            bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 60)
            ax.set_xscale("log")
        else:
            values = raw_values
            bins = 60
        ax.hist(values, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.3)

        x_min, x_max = values.min(), values.max()
        drawn_low = max(low, log_floor) if metric in log_scale_metrics else low
        if drawn_low > x_min:
            ax.axvspan(x_min, drawn_low, color="#D62728", alpha=0.15)
        if high < x_max:
            ax.axvspan(high, x_max, color="#D62728", alpha=0.15)
        ax.axvline(drawn_low, color="#D62728", linestyle="--",
                   label=f"low cutoff {_format_cutoff(low)} ({n_low:,} images)")
        ax.axvline(high, color="#D62728", linestyle=":",
                   label=f"high cutoff {_format_cutoff(high)} ({n_high:,} images)")
        ax.set_xlim(min(x_min, drawn_low), max(x_max, high))
        ax.set_title(metric.capitalize() + (" (log scale)" if metric in log_scale_metrics else ""))
        ax.set_xlabel(metric)
        ax.set_ylabel("Number of images")
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_metrics_by_jpeg_quality(
    dataframe,
    cutoffs,
    metrics=("sharpness", "noise", "blockiness", "pixelation"),
    log_scale_metrics=("sharpness", "noise", "pixelation"),
    log_floor=0.01,
):
    """Bar chart of JPEG save quality, then each metric's histogram split by that quality.

    Each quality level gets its own colour, drawn on shared bins so the groups
    can be compared directly. The dotted line is the metric's high cutoff and
    the dashed line its low cutoff (drawn only for metrics flagged on the low side).
    """
    quality = dataframe["jpeg_quality"].round()
    levels = sorted(quality.dropna().unique())
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    colors = {level: palette[i % len(palette)] for i, level in enumerate(levels)}

    n_panels = 1 + len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))

    counts = quality.value_counts().reindex(levels)
    bars = axes[0].bar([f"q{int(level)}" for level in levels], counts.values,
                       color=[colors[level] for level in levels])
    for bar, count in zip(bars, counts.values):
        axes[0].annotate(f"{count:,}\n({100 * count / len(dataframe):.0f}%)",
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, counts.max() * 1.2)
    axes[0].set_title("JPEG save quality")
    axes[0].set_ylabel("Number of images")

    for ax, metric in zip(axes[1:], metrics):
        values = dataframe[metric]
        is_log = metric in log_scale_metrics
        if is_log:
            values = values.clip(lower=log_floor)
            valid = values.dropna()
            bins = np.logspace(np.log10(valid.min()), np.log10(valid.max()), 60)
            ax.set_xscale("log")
        else:
            bins = np.linspace(values.min(), values.max(), 60)
        for level in levels:
            group = values[quality == level].dropna()
            ax.hist(group, bins=bins, color=colors[level], alpha=0.6,
                    label=f"q{int(level)} ({len(group):,})")
        low, high = cutoffs[metric]
        if metric == "sharpness":
            ax.axvline(low, color="#D62728", linestyle="--", label=f"low cutoff {_format_cutoff(low)}")
        else:
            ax.axvline(high, color="#D62728", linestyle=":", label=f"high cutoff {_format_cutoff(high)}")
        ax.set_title(metric.capitalize() + (" (log scale)" if is_log else "") + " by JPEG quality")
        ax.set_xlabel(metric)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def show_outliers(df_subset, title, metric=None, n_cols=8, images_dir=IMAGES_DIR):
    """Show images in a grid with filename, class and (optionally) one metric value."""
    if df_subset.empty:
        print(f"No images to display for: {title}")
        return

    n_rows = int(np.ceil(len(df_subset) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, df_subset.iterrows()):
        with Image.open(Path(images_dir) / str(row["filename"])) as image:
            ax.imshow(image.convert("RGB"))
        caption = f"{row['filename']}\n{str(row['description'])[:16]}"
        if metric is not None:
            caption += f"\n{metric}: {_format_cutoff(row[metric])}"
        ax.set_title(caption, fontsize=7)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    plt.show()


class TinyImageNetEvalDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.images_dir / str(row["filename"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, row["class_id"], row["description"], row["filename"]


def compute_label_noise_scores(embeddings, dataframe, k=15):
    if k <= 0 or len(dataframe) <= k:
        raise ValueError(f"need more than {k} samples and a positive neighbor count")
    if len(dataframe) != len(embeddings):
        raise ValueError("dataframe rows and embeddings must have equal lengths")

    nn_model = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",
    )
    nn_model.fit(embeddings)
    indices = nn_model.kneighbors(embeddings, return_distance=False)

    class_list = dataframe["class_id"].to_numpy()
    desc_list = dataframe["description"].to_numpy()
    scores = []
    suggested_classes = []
    suggested_descs = []
    consensus_ratios = []

    for index in range(len(dataframe)):
        neighbor_indices = indices[index, 1:]
        neighbor_classes = class_list[neighbor_indices]
        neighbor_descs = desc_list[neighbor_indices]
        current_class = class_list[index]

        match_count = np.sum(neighbor_classes == current_class)
        scores.append(1.0 - (match_count / k))

        values, counts = np.unique(neighbor_classes, return_counts=True)
        top_class = values[np.argmax(counts)]
        suggested_classes.append(top_class)
        suggested_descs.append(neighbor_descs[neighbor_classes == top_class][0])
        consensus_ratios.append(counts.max() / k)

    result = dataframe.copy()
    result["inconsistency_score"] = scores
    result["suggested_class"] = suggested_classes
    result["suggested_desc"] = suggested_descs
    result["consensus_ratio"] = consensus_ratios
    return result, indices[:, 1:]


def inspect_top_suspicious_cases(
    audit_df,
    neighbor_map,
    images_dir=IMAGES_DIR,
    top_n=10,
    k_show=4,
):
    candidates = audit_df[
        (audit_df["inconsistency_score"] >= 0.90)
        & (audit_df["consensus_ratio"] >= 0.70)
    ].sort_values(
        by=["inconsistency_score", "consensus_ratio"],
        ascending=False,
    ).head(top_n)

    if candidates.empty:
        print("No images found with inconsistency score >= 0.90 and consensus ratio >= 0.70.")
        return

    images_dir = Path(images_dir)
    for rank, (original_index, row) in enumerate(candidates.iterrows(), 1):
        fig, axes = plt.subplots(1, k_show + 1, figsize=(14, 3))

        with Image.open(images_dir / str(row["filename"])) as image:
            axes[0].imshow(image.convert("RGB"))
        axes[0].set_title(
            f"CASE #{rank} (SUSPICIOUS)\nFile: {row['filename']}\n"
            f"Assigned: {row['description'][:18]}\n"
            f"Score: {row['inconsistency_score']:.2f}",
            color="red",
            fontsize=8,
            fontweight="bold",
        )
        axes[0].axis("off")

        nearest_neighbors = neighbor_map[original_index][:k_show]
        for axis_index, neighbor_index in enumerate(nearest_neighbors, 1):
            neighbor = audit_df.iloc[neighbor_index]
            with Image.open(images_dir / str(neighbor["filename"])) as image:
                axes[axis_index].imshow(image.convert("RGB"))
            axes[axis_index].set_title(
                f"Neighbor #{axis_index}\n{neighbor['description'][:18]}\n"
                f"(ID: {neighbor['class_id']})",
                color="navy",
                fontsize=8,
            )
            axes[axis_index].axis("off")

        fig.suptitle(
            f"Audit Flag: Labeled as [{row['description']}] | "
            f"Neighbor Consensus: [{row['suggested_desc']}] "
            f"(Agreement: {row['consensus_ratio'] * 100:.0f}%)",
            fontsize=10,
            y=1.05,
        )
        fig.tight_layout()
        plt.show()
