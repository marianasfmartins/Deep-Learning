"""Train-fitted preprocessing steps for Tiny ImageNet (run after the split).

Both steps learn only from the training split and then treat every split the same way:

* ``MislabelFilter``: ``fit(train)`` builds a nearest-neighbour index over the training
  images; ``apply(split)`` returns the split without the suspected mislabeled images.
* ``ImageQualityPipeline``: ``fit(train)`` computes the IQR thresholds and calibrates the
  fix strengths; ``transform(split, ...)`` writes the restored images and returns the split.

Deduplication happens before the split (``preprocessing_utils.remove_duplicates``).
"""

from collections import Counter
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from preprocessing_utils import (
    BASE,
    RGB_IMAGES_DIR,
    QUALITY_MEASURES,
    SEED,
    add_gaussian_blur,
    add_gaussian_noise,
    adjust_gamma,
    compress_contrast,
    deblock_image,
    deblur_image,
    denoise_image,
    depixelate_image,
    enhance_contrast,
    iqr_cutoffs,
    load_embeddings,
    measure_quality,
    psnr,
    read_rgb,
    write_rgb,
)


class MislabelFilter:
    """Remove images whose nearest training neighbours reject their label.

    Each image is compared with its ``k`` nearest **training** images (cosine distance
    between ResNet-18 embeddings saved in Section 1.9.4). It is flagged when at least
    ``min_disagree`` of them carry a different label (S >= 14/15 = 0.93) and at least
    ``min_consensus`` agree on one other class (C >= 11/15, the "C >= 0.70" rule).
    Only training labels are used, so validation and test can be filtered without
    using their own labels' neighbourhoods or any model result.
    """

    def __init__(self, embeddings_path=BASE / "embeddings_resnet18.npz", k=15,
                 min_disagree=14, min_consensus=11):
        self.embeddings_path = Path(embeddings_path)
        self.k = k
        self.min_disagree = min_disagree
        self.min_consensus = min_consensus

    def fit(self, train):
        """Build the neighbour index over the training images and store their labels."""
        self.train_filenames_ = train["filename"].astype(str).to_numpy()
        self.train_classes_ = train["class_id"].astype(str).to_numpy()
        embeddings = load_embeddings(self.embeddings_path, self.train_filenames_)
        self.index_ = NearestNeighbors(n_neighbors=self.k + 1, metric="cosine", algorithm="brute")
        self.index_.fit(embeddings)
        return self

    def score(self, split):
        """S, C, the suggested class and the flag for every image of ``split``."""
        filenames = split["filename"].astype(str).to_numpy()
        classes = split["class_id"].astype(str).to_numpy()
        embeddings = load_embeddings(self.embeddings_path, filenames)
        neighbour_rows = self.index_.kneighbors(embeddings, return_distance=False)

        rows = []
        for filename, own_class, candidates in zip(filenames, classes, neighbour_rows):
            # A training image is its own nearest neighbour: skip it, keep k others.
            neighbours = [row for row in candidates if self.train_filenames_[row] != filename][: self.k]
            neighbour_classes = self.train_classes_[neighbours]
            n_disagree = int((neighbour_classes != own_class).sum())
            other_counts = Counter(neighbour_classes[neighbour_classes != own_class])
            suggested, n_consensus = other_counts.most_common(1)[0] if other_counts else (own_class, 0)
            rows.append({
                "filename": filename,
                "inconsistency_score": n_disagree / self.k,
                "consensus_ratio": n_consensus / self.k,
                "suggested_class": suggested,
                "suspected_mislabel": n_disagree >= self.min_disagree and n_consensus >= self.min_consensus,
            })
        return pd.DataFrame(rows, index=split.index)

    def split_flagged(self, split):
        """Return ``(kept, removed)``: the split without the flagged images, and the flagged
        images with their scores and suggested class."""
        scores = self.score(split)
        flagged = scores["suspected_mislabel"]
        removed = split.loc[flagged].join(
            scores.loc[flagged, ["inconsistency_score", "consensus_ratio", "suggested_class"]])
        return split.loc[~flagged].reset_index(drop=True), removed.reset_index(drop=True)

    def apply(self, split):
        """Return a copy of ``split`` without the flagged images."""
        return self.split_flagged(split)[0]


# flag name: (measure, side of the cutoff, cutoff on a log scale)
QUALITY_FLAGS = {
    "very_dark": ("brightness", "low", False),
    "very_bright": ("brightness", "high", False),
    "low_contrast": ("contrast", "low", False),
    "high_contrast": ("contrast", "high", False),
    "blurry": ("sharpness", "low", True),
    "noisy": ("noise", "high", True),
    "compression_blocks": ("blockiness", "high", False),
    "pixelated": ("pixelation", "high", True),
}


class ImageQualityPipeline:
    """Flag and restore low-quality images with thresholds fitted on the training split.

    ``fit(train)``:
      * measures the six quality scores of every training image;
      * computes the IQR (boxplot) cutoffs from the training scores only;
      * sets the brightness and contrast targets to the median of unflagged training images;
      * calibrates the deblurring and denoising strengths: clean training images are
        degraded on purpose until they match the typical flagged image, and the strength
        that restores them best (highest PSNR against the clean original) is kept.

    ``transform(split, ...)`` flags each image with the fitted cutoffs, applies the
    selected fixes, and writes every image of the split to ``output_dir`` under its
    original filename (restored images are re-saved, unchanged ones copied). It returns
    the split with one column per flag and a ``fixes_applied`` column.

    ``images_dir`` must hold RGB images: the output of ``convert_to_rgb`` (before the split).
    """

    def __init__(
        self,
        images_dir=RGB_IMAGES_DIR,
        n_calibration=100,
        blur_sigmas=tuple(np.arange(0.5, 6.01, 0.25)),
        deblur_iterations=(10, 20, 50, 100),
        noise_sigmas=tuple(range(2, 61, 2)),
        denoise_strengths=(3, 5, 7, 10, 15, 20, 25),
        clahe_clip_limit=2.0,
        deblock_params=(5, 25, 5),
        depixelate_kernel=3,
        n_workers=8,
        seed=SEED,
    ):
        self.images_dir = Path(images_dir)
        self.n_calibration = n_calibration
        self.blur_sigmas = blur_sigmas
        self.deblur_iterations = deblur_iterations
        self.noise_sigmas = noise_sigmas
        self.denoise_strengths = denoise_strengths
        self.clahe_clip_limit = clahe_clip_limit
        self.deblock_params = deblock_params
        self.depixelate_kernel = depixelate_kernel
        self.n_workers = n_workers
        self.seed = seed

    # ---- measuring and flagging -------------------------------------------------

    def _map(self, function, items):
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            return list(pool.map(function, items))

    def measure(self, filenames, images_dir=None):
        """The six quality scores of each image, indexed by filename (from ``images_dir``,
        by default the pipeline's input folder)."""
        images_dir = self.images_dir if images_dir is None else Path(images_dir)
        filenames = [str(name) for name in filenames]
        scores = self._map(lambda name: measure_quality(read_rgb(images_dir / name)), filenames)
        return pd.DataFrame(scores, index=pd.Index(filenames, name="filename"))[QUALITY_MEASURES]

    def flag(self, scores):
        """One True/False column per quality flag, using the fitted cutoffs."""
        flags = {}
        for flag, (measure, side, _) in QUALITY_FLAGS.items():
            low, high = self.cutoffs_[measure]
            flags[flag] = scores[measure] < low if side == "low" else scores[measure] > high
        return pd.DataFrame(flags, index=scores.index)

    # ---- fitting ----------------------------------------------------------------

    def fit(self, train):
        """Fit the cutoffs, targets and fix strengths on the training split."""
        self.train_scores_ = self.measure(train["filename"])
        log_scale = {measure: is_log for measure, _, is_log in QUALITY_FLAGS.values()}
        self.cutoffs_ = {
            measure: iqr_cutoffs(self.train_scores_[measure], log_scale=log_scale.get(measure, False))
            for measure in QUALITY_MEASURES
        }
        train_flags = self.flag(self.train_scores_)
        unflagged = self.train_scores_[~train_flags.any(axis=1)]
        self.targets_ = {
            "brightness": float(unflagged["brightness"].median()),
            "contrast": float(unflagged["contrast"].median()),
        }

        # Calibration references: clean RGB training images (no flag, not grayscale)
        candidates = unflagged[unflagged["noise"] > 0].index
        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(candidates, size=min(self.n_calibration, len(candidates)), replace=False)
        clean = self._map(lambda name: read_rgb(self.images_dir / name), chosen)

        blur_target = float(self.train_scores_.loc[train_flags["blurry"], "sharpness"].median())
        noise_target = float(self.train_scores_.loc[train_flags["noisy"], "noise"].median())
        deblur_row = self._calibrate_deblur(clean, blur_target)
        denoise_row = self._calibrate_denoise(clean, noise_target, rng)
        self.calibration_ = pd.DataFrame([deblur_row, denoise_row]).set_index("fix")
        self.train_flag_counts_ = train_flags.sum()
        return self

    def _closest_degradation(self, clean, degrade, levels, measure, target):
        """The degradation level whose median ``measure`` is closest to ``target`` (log scale)."""
        def median_score(level):
            return float(np.median([measure_quality(degrade(image, level))[measure] for image in clean]))
        medians = {level: median_score(level) for level in levels}
        best = min(medians, key=lambda level: abs(np.log10(medians[level] + 1e-6) - np.log10(target + 1e-6)))
        return best, medians[best]

    def _best_strength(self, clean, degraded, restore, strengths):
        """Mean PSNR before restoring and for each strength; return the best strength."""
        psnr_degraded = float(np.mean([psnr(c, d) for c, d in zip(clean, degraded)]))
        psnr_by_strength = {
            strength: float(np.mean(self._map(lambda pair: psnr(pair[0], restore(pair[1], strength)),
                                              list(zip(clean, degraded)))))
            for strength in strengths
        }
        best = max(psnr_by_strength, key=psnr_by_strength.get)
        return best, psnr_degraded, psnr_by_strength

    def _calibrate_deblur(self, clean, target_sharpness):
        sigma, matched = self._closest_degradation(
            clean, add_gaussian_blur, self.blur_sigmas, "sharpness", target_sharpness)
        blurred = [add_gaussian_blur(image, sigma) for image in clean]
        iterations, psnr_blurred, psnr_by_iterations = self._best_strength(
            clean, blurred, lambda image, n: deblur_image(image, sigma, n), self.deblur_iterations)
        self.deblur_sigma_, self.deblur_iterations_ = float(sigma), int(iterations)
        return {
            "fix": "deblur (Richardson-Lucy)",
            "flagged median": f"sharpness {target_sharpness:,.1f}",
            "synthetic degradation": f"Gaussian blur sigma {sigma:.2f} (sharpness {matched:,.1f})",
            "chosen strength": f"PSF sigma {sigma:.2f}, {iterations} iterations",
            "PSNR degraded (dB)": round(psnr_blurred, 2),
            "PSNR restored (dB)": round(psnr_by_iterations[iterations], 2),
            "PSNR by strength": {k: round(v, 2) for k, v in psnr_by_iterations.items()},
        }

    def _calibrate_denoise(self, clean, target_noise, rng):
        def add_noise(image, sigma):
            return add_gaussian_noise(image, sigma, np.random.default_rng(self.seed))
        sigma, matched = self._closest_degradation(
            clean, add_noise, self.noise_sigmas, "noise", target_noise)
        noisy = [add_gaussian_noise(image, sigma, rng) for image in clean]
        strength, psnr_noisy, psnr_by_strength = self._best_strength(
            clean, noisy, denoise_image, self.denoise_strengths)
        self.denoise_strength_ = float(strength)
        return {
            "fix": "denoise (non-local means)",
            "flagged median": f"noise {target_noise:,.2f}",
            "synthetic degradation": f"Gaussian noise sigma {sigma} (noise {matched:,.2f})",
            "chosen strength": f"h = {strength}",
            "PSNR degraded (dB)": round(psnr_noisy, 2),
            "PSNR restored (dB)": round(psnr_by_strength[strength], 2),
            "PSNR by strength": {k: round(v, 2) for k, v in psnr_by_strength.items()},
        }

    def thresholds_table(self):
        """The fitted cutoffs and how many training images each flag catches."""
        rows = []
        for flag, (measure, side, is_log) in QUALITY_FLAGS.items():
            low, high = self.cutoffs_[measure]
            rows.append({
                "flag": flag,
                "rule": f"{measure} {'<' if side == 'low' else '>'} {low if side == 'low' else high:,.2f}",
                "log-scale IQR": is_log,
                "training images flagged": int(self.train_flag_counts_[flag]),
            })
        return pd.DataFrame(rows)

    # ---- transforming -----------------------------------------------------------

    def _restore(self, rgb, scores, flags, enabled):
        """Apply the enabled fixes this image is flagged for, in a fixed order."""
        applied = []
        if enabled["denoise"] and flags["noisy"]:            # before deblurring: it would amplify noise
            rgb = denoise_image(rgb, self.denoise_strength_); applied.append("denoise")
        if enabled["deblock"] and flags["compression_blocks"]:
            rgb = deblock_image(rgb, *self.deblock_params); applied.append("deblock")
        if enabled["depixelate"] and flags["pixelated"]:
            rgb = depixelate_image(rgb, self.depixelate_kernel); applied.append("depixelate")
        if enabled["deblur"] and flags["blurry"]:
            rgb = deblur_image(rgb, self.deblur_sigma_, self.deblur_iterations_); applied.append("deblur")
        if enabled["fix_brightness"] and (flags["very_dark"] or flags["very_bright"]):
            rgb = adjust_gamma(rgb, scores["brightness"], self.targets_["brightness"]); applied.append("gamma")
        if enabled["fix_contrast"] and flags["low_contrast"]:
            rgb = enhance_contrast(rgb, self.clahe_clip_limit); applied.append("clahe")
        if enabled["fix_contrast"] and flags["high_contrast"]:
            rgb = compress_contrast(rgb, scores["contrast"], self.targets_["contrast"]); applied.append("compress contrast")
        return rgb, applied

    def transform(self, split, output_dir, deblur=True, denoise=True, deblock=False,
                  depixelate=False, fix_brightness=False, fix_contrast=False, jpeg_quality=95):
        """Restore the flagged images of ``split`` and save all its images to ``output_dir``.

        Deblurring and denoising are on by default. The other fixes are off by default
        because the EDA (1.8.6) found those flags to be mostly natural images (dark,
        bright, plain backgrounds) or false alarms (pixelation).
        """
        enabled = {"deblur": deblur, "denoise": denoise, "deblock": deblock,
                   "depixelate": depixelate, "fix_brightness": fix_brightness,
                   "fix_contrast": fix_contrast}
        output_dir = Path(output_dir)
        if output_dir.resolve() == self.images_dir.resolve():
            raise ValueError("output_dir must differ from images_dir, so a re-run never restores an image twice")
        output_dir.mkdir(parents=True, exist_ok=True)
        filenames = split["filename"].astype(str).tolist()
        known = self.train_scores_.reindex(filenames)
        unknown = known.index[known.isna().any(axis=1)]
        if len(unknown):
            known.loc[unknown] = self.measure(unknown).to_numpy()
        flags = self.flag(known)

        def process(name):
            rgb, applied = self._restore(read_rgb(self.images_dir / name), known.loc[name], flags.loc[name], enabled)
            if applied:
                write_rgb(output_dir / name, rgb, jpeg_quality)
            else:  # unchanged: copy the file, so it is not re-compressed
                shutil.copyfile(self.images_dir / name, output_dir / name)
            return "; ".join(applied)

        result = split.copy().reset_index(drop=True)
        for measure in QUALITY_MEASURES:
            result[f"{measure}_before"] = known[measure].to_numpy()
        for flag in QUALITY_FLAGS:
            result[f"flag_{flag}"] = flags[flag].to_numpy()
        result["fixes_applied"] = self._map(process, filenames)
        return result

    def evaluate(self, transformed, output_dir):
        """Compare the restored images with themselves before the fixes.

        Re-measures every image that ``transform`` changed (unchanged images are exact
        copies, so their scores cannot move) and flags it again with the same fitted
        cutoffs. Returns one row per changed image with ``<measure>_before``,
        ``<measure>_after``, and ``<flag>_before`` / ``<flag>_after`` columns.
        """
        changed = transformed.loc[transformed["fixes_applied"] != ""].reset_index(drop=True)
        before = changed.set_index("filename")[[f"{m}_before" for m in QUALITY_MEASURES]]
        before.columns = QUALITY_MEASURES
        after = self.measure(changed["filename"], images_dir=output_dir)
        flags_before, flags_after = self.flag(before), self.flag(after)

        result = changed[["filename", "class_id", "description", "fixes_applied"]].copy()
        for measure in QUALITY_MEASURES:
            result[f"{measure}_before"] = before[measure].to_numpy()
            result[f"{measure}_after"] = after[measure].to_numpy()
        for flag in QUALITY_FLAGS:
            result[f"{flag}_before"] = flags_before[flag].to_numpy()
            result[f"{flag}_after"] = flags_after[flag].to_numpy()
        return result
