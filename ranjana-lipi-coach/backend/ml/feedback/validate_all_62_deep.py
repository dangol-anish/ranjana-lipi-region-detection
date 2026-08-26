#!/usr/bin/env python3
"""Deep validation suite for the full 62-class Ranjana Lipi pipeline.

This script does not train or modify models. It verifies the already-trained
general recognizer and per-character autoencoders with repeatable checks:

1. Asset coverage for every dataset class.
2. Good-sample sanity checks on real processed_general samples.
3. Synthetic broad-band flaw checks by erasing top/middle/bottom ink bands.
4. Wrong-character blocking checks through the actual upload-style pipeline.
5. A compact proof image and CSV/JSON outputs for final-defense documentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.feedback.grid_feedback import DEFAULT_INK_THRESHOLD, build_region_feedback  # noqa: E402
from ml.feedback.template_feedback import StrokeTemplate, load_template  # noqa: E402
from ml.inference.pipeline import (  # noqa: E402
    analyze_attempt,
    load_general_autoencoder,
    load_general_recognizer,
    load_general_region_baseline,
    normalized_ink_pixel_count,
    recognize_general,
    reconstruct_and_feedback,
    tensor_from_normalized,
)
from ml.training.dataset import CLASSES as VALIDATED_CLASSES  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
BANDS = ("top", "middle", "bottom")
VALIDATED_CLASS_SET = set(VALIDATED_CLASSES)


@dataclass(frozen=True)
class ClassAssets:
    class_name: str
    processed_dir: Path
    augmented_dir: Path
    reference_path: Path
    autoencoder_path: Path
    baseline_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-validate all 62 Ranjana Lipi classes")
    parser.add_argument("--good-samples-per-class", type=int, default=5)
    parser.add_argument("--synthetic-samples-per-band", type=int, default=1)
    parser.add_argument(
        "--synthetic-flaw-kind",
        choices=("extra_stroke", "missing_stroke"),
        default="extra_stroke",
        help=(
            "extra_stroke adds a controlled false stroke in the intended band; "
            "missing_stroke erases existing ink in the intended band."
        ),
    )
    parser.add_argument("--wrong-character-samples-per-class", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Final_demo_images" / "10_full_62_deep_validation",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def dataset_classes() -> list[str]:
    dataset_root = PROJECT_ROOT / "data" / "Dataset"
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def processed_general_root() -> Path:
    return BACKEND_ROOT / "ml" / "processed_general"


def saved_models_root() -> Path:
    return BACKEND_ROOT / "ml" / "saved_models"


def class_assets(class_name: str) -> ClassAssets:
    saved = saved_models_root()
    return ClassAssets(
        class_name=class_name,
        processed_dir=processed_general_root() / class_name,
        augmented_dir=BACKEND_ROOT / "ml" / "augmented_general" / class_name,
        reference_path=processed_general_root() / "references" / f"{class_name}.png",
        autoencoder_path=saved / "autoencoders_general" / f"autoencoder_{class_name}.pt",
        baseline_path=saved / "autoencoders_general" / f"region_baseline_{class_name}.json",
    )


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_normalized(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return (image.astype(np.float32) / 255.0).clip(0.0, 1.0)


def encode_png_bytes(normalized_or_gray: np.ndarray) -> bytes:
    array = np.asarray(normalized_or_gray)
    if array.dtype != np.uint8:
        array = (array.clip(0.0, 1.0) * 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise ValueError("Could not encode PNG bytes")
    return bytes(encoded)


def choose_good_samples(class_name: str, count: int) -> list[Path]:
    files = image_files(processed_general_root() / class_name)
    selected: list[Path] = []
    for path in files:
        normalized = read_normalized(path)
        if normalized_ink_pixel_count(normalized) >= 300:
            selected.append(path)
        if len(selected) >= count:
            break
    return selected


def choose_clean_template_samples(
    class_name: str,
    count: int,
    device: torch.device,
    candidate_limit: int = 25,
) -> list[Path]:
    candidates = choose_good_samples(class_name, candidate_limit)
    scored: list[tuple[float, Path]] = []
    for path in candidates:
        normalized = read_normalized(path)
        feedback = run_autoencoder_feedback(normalized, class_name, device)
        cleanliness_score = max(
            max_region_score(feedback, "fine_grid"),
            max_region_score(feedback, "broad_bands"),
        )
        scored.append((cleanliness_score, path))
    return [path for _score, path in sorted(scored, key=lambda item: item[0])[:count]]


def erode_band(normalized: np.ndarray, band: str) -> np.ndarray:
    flawed = np.asarray(normalized, dtype=np.float32).copy()
    height, _width = flawed.shape
    band_index = BANDS.index(band)
    y0 = round(band_index * height / 3)
    y1 = round((band_index + 1) * height / 3)
    band_region = flawed[y0:y1, :]
    ink_mask = band_region > DEFAULT_INK_THRESHOLD
    if np.count_nonzero(ink_mask) == 0:
        return flawed

    ys, xs = np.where(ink_mask)
    local_y0, local_y1 = int(ys.min()), int(ys.max()) + 1
    local_x0, local_x1 = int(xs.min()), int(xs.max()) + 1
    box_h = max(1, local_y1 - local_y0)
    box_w = max(1, local_x1 - local_x0)

    erase_y0 = y0 + local_y0 + box_h // 4
    erase_y1 = y0 + local_y0 + (box_h * 3) // 4
    erase_x0 = local_x0 + box_w // 5
    erase_x1 = local_x0 + (box_w * 4) // 5
    flawed[erase_y0:erase_y1, erase_x0:erase_x1] = 0.0
    return flawed


def remove_expected_template_band(
    normalized: np.ndarray,
    band: str,
    template: StrokeTemplate,
    expected_threshold: float = 0.08,
) -> np.ndarray:
    flawed = np.asarray(normalized, dtype=np.float32).copy()
    height, _width = flawed.shape
    band_index = BANDS.index(band)
    y0 = round(band_index * height / 3)
    y1 = round((band_index + 1) * height / 3)
    expected_band = template.mean_ink_map[y0:y1, :] > expected_threshold
    if np.count_nonzero(expected_band) == 0:
        return erode_band(normalized, band)
    flawed[y0:y1, :][expected_band] = 0.0
    return flawed


def add_extra_stroke_to_band(normalized: np.ndarray, band: str) -> np.ndarray:
    flawed = np.asarray(normalized, dtype=np.float32).copy()
    height, width = flawed.shape
    band_index = BANDS.index(band)
    y0 = round(band_index * height / 3)
    y1 = round((band_index + 1) * height / 3)
    band_height = max(1, y1 - y0)

    stroke = np.zeros_like(flawed, dtype=np.uint8)
    y_mid = y0 + band_height // 2
    x_start = max(6, width // 5)
    x_end = min(width - 7, (width * 4) // 5)
    slant = max(3, band_height // 5)
    cv2.line(
        stroke,
        (x_start, max(y0 + 4, y_mid - slant)),
        (x_end, min(y1 - 5, y_mid + slant)),
        color=255,
        thickness=9,
        lineType=cv2.LINE_AA,
    )
    cv2.line(
        stroke,
        (x_start, min(y1 - 5, y_mid + slant)),
        (x_end, max(y0 + 4, y_mid - slant)),
        color=255,
        thickness=7,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(stroke, (x_start + 10, y_mid), 7, color=255, thickness=-1, lineType=cv2.LINE_AA)
    flawed = np.maximum(flawed, stroke.astype(np.float32) / 255.0)
    return flawed


def make_synthetic_flaw(
    normalized: np.ndarray,
    band: str,
    flaw_kind: str,
    template: StrokeTemplate | None = None,
) -> np.ndarray:
    if flaw_kind == "missing_stroke":
        if template is not None:
            return remove_expected_template_band(normalized, band, template)
        return erode_band(normalized, band)
    return add_extra_stroke_to_band(normalized, band)


def run_autoencoder_feedback(
    normalized: np.ndarray,
    class_name: str,
    device: torch.device,
) -> dict[str, Any]:
    if class_name in VALIDATED_CLASS_SET:
        return reconstruct_and_feedback(normalized, class_name, device_name=str(device))

    model = load_general_autoencoder(class_name, str(device))
    baseline = load_general_region_baseline(class_name)
    image_tensor = tensor_from_normalized(normalized).to(device)
    with torch.no_grad():
        reconstruction = model(image_tensor).cpu().squeeze(0).squeeze(0).numpy()
    return build_region_feedback(
        class_name=class_name,
        input_image=normalized,
        reconstruction=reconstruction,
        rows=3,
        cols=3,
        max_regions=3,
        min_problem_region_error=0.012,
        baseline=baseline,
    )


def top_region(feedback: dict[str, Any], group: str) -> str:
    regions = feedback.get(group, {}).get("all_regions", [])
    return str(regions[0]["region"]) if regions else ""


def problem_count(feedback: dict[str, Any], group: str) -> int:
    return len(feedback.get(group, {}).get("problem_regions", []))


def max_region_score(feedback: dict[str, Any], group: str) -> float:
    group_feedback = feedback.get(group, {})
    if "max_z_score" in group_feedback:
        return float(group_feedback["max_z_score"])
    regions = group_feedback.get("all_regions", [])
    if not regions:
        return 0.0
    return max(float(region.get("adjusted_score", region.get("z_score", region.get("score", 0.0)))) for region in regions)


def validate_assets(classes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_name in classes:
        assets = class_assets(class_name)
        processed_count = len(image_files(assets.processed_dir))
        augmented_count = len(image_files(assets.augmented_dir))
        row = {
            "class": class_name,
            "processed_images": processed_count,
            "augmented_images": augmented_count,
            "reference_exists": assets.reference_path.is_file(),
            "autoencoder_exists": assets.autoencoder_path.is_file() if class_name not in VALIDATED_CLASS_SET else (saved_models_root() / f"autoencoder_{class_name}.pt").is_file(),
            "baseline_exists": assets.baseline_path.is_file() if class_name not in VALIDATED_CLASS_SET else (saved_models_root() / f"region_baseline_{class_name}.json").is_file(),
        }
        row["asset_pass"] = all(
            [
                processed_count > 0,
                augmented_count > 0,
                row["reference_exists"],
                row["autoencoder_exists"],
                row["baseline_exists"],
            ]
        )
        rows.append(row)
    return rows


def validate_good_samples(
    classes: list[str],
    samples_per_class: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_name in classes:
        for path in choose_good_samples(class_name, samples_per_class):
            normalized = read_normalized(path)
            recognizer = recognize_general(normalized, str(device))
            feedback = run_autoencoder_feedback(normalized, class_name, device)
            fine_count = problem_count(feedback, "fine_grid")
            broad_count = problem_count(feedback, "broad_bands")
            rows.append(
                {
                    "class": class_name,
                    "filename": path.name,
                    "path": str(path),
                    "predicted_class": recognizer.predicted_class,
                    "recognizer_confidence": recognizer.confidence,
                    "recognizer_match": recognizer.predicted_class == class_name,
                    "overall_score": feedback["overall_score"],
                    "fine_problem_count": fine_count,
                    "broad_problem_count": broad_count,
                    "max_fine_z": max_region_score(feedback, "fine_grid"),
                    "max_broad_z": max_region_score(feedback, "broad_bands"),
                    "good_pass": recognizer.predicted_class == class_name and fine_count == 0 and broad_count == 0,
                }
            )
    return rows


def validate_synthetic_flaws(
    classes: list[str],
    samples_per_band: int,
    device: torch.device,
    output_dir: Path,
    flaw_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    synthetic_dir = output_dir / "synthetic_flaws"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    for class_name in classes:
        template = load_template(BACKEND_ROOT / "ml" / "saved_models" / "stroke_templates" / f"{class_name}.npz")
        samples = choose_clean_template_samples(class_name, max(1, samples_per_band), device)
        for sample_index, path in enumerate(samples[:samples_per_band], start=1):
            normalized = read_normalized(path)
            for band in BANDS:
                flawed = make_synthetic_flaw(normalized, band, flaw_kind, template)
                synthetic_path = synthetic_dir / f"{class_name}_{flaw_kind}_{band}_{sample_index:02d}.png"
                Image.fromarray((flawed * 255).astype(np.uint8), mode="L").save(synthetic_path)
                feedback = run_autoencoder_feedback(flawed, class_name, device)
                broad_top = top_region(feedback, "broad_bands")
                fine_top = top_region(feedback, "fine_grid")
                broad_top_2 = [
                    str(region["region"])
                    for region in feedback.get("broad_bands", {}).get("all_regions", [])[:2]
                ]
                rows.append(
                    {
                        "class": class_name,
                        "source_filename": path.name,
                        "synthetic_file": str(synthetic_path),
                        "flaw_kind": flaw_kind,
                        "intended_band": band,
                        "broad_top_1": broad_top,
                        "broad_top_2": ";".join(broad_top_2),
                        "fine_top_1": fine_top,
                        "overall_score": feedback["overall_score"],
                        "max_broad_z": max_region_score(feedback, "broad_bands"),
                        "broad_problem_count": problem_count(feedback, "broad_bands"),
                        "exact_match": broad_top == band,
                        "top2_match": band in broad_top_2,
                    }
                )
    return rows


def validate_wrong_character_blocking(
    classes: list[str],
    samples_per_class: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, class_name in enumerate(classes):
        wrong_target = classes[(index + 1) % len(classes)]
        for path in choose_good_samples(class_name, samples_per_class):
            # This intentionally exercises the same uploaded-image code path as
            # FastAPI: bytes -> target-class normalization -> recognizer gate.
            analysis = analyze_attempt(
                image_bytes=encode_png_bytes(read_normalized(path)),
                target_class=wrong_target,
            )
            feedback = analysis["feedback"]
            rows.append(
                {
                    "source_class": class_name,
                    "wrong_target_class": wrong_target,
                    "filename": path.name,
                    "wrong_character": bool(feedback.get("wrong_character")),
                    "overall_score": feedback.get("overall_score"),
                    "predicted_class": feedback.get("predicted_class"),
                    "recognizer_confidence": feedback.get("recognizer_confidence"),
                    "blocking_pass": bool(feedback.get("wrong_character")) and feedback.get("overall_score") == 0.0,
                }
            )
            break
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(count: int, total: int) -> float:
    return 0.0 if total == 0 else count * 100.0 / total


def summarize(
    classes: list[str],
    asset_rows: list[dict[str, Any]],
    good_rows: list[dict[str, Any]],
    flaw_rows: list[dict[str, Any]],
    wrong_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    asset_pass = sum(bool(row["asset_pass"]) for row in asset_rows)
    good_pass = sum(bool(row["good_pass"]) for row in good_rows)
    flaw_exact = sum(bool(row["exact_match"]) for row in flaw_rows)
    flaw_top2 = sum(bool(row["top2_match"]) for row in flaw_rows)
    wrong_pass = sum(bool(row["blocking_pass"]) for row in wrong_rows)
    return {
        "classes_total": len(classes),
        "assets_pass": asset_pass,
        "assets_pass_rate": pct(asset_pass, len(asset_rows)),
        "good_samples_total": len(good_rows),
        "good_samples_pass": good_pass,
        "good_samples_pass_rate": pct(good_pass, len(good_rows)),
        "good_samples_any_fine_flag": sum(row["fine_problem_count"] > 0 for row in good_rows),
        "good_samples_any_broad_flag": sum(row["broad_problem_count"] > 0 for row in good_rows),
        "synthetic_flaws_total": len(flaw_rows),
        "synthetic_flaws_broad_exact": flaw_exact,
        "synthetic_flaws_broad_exact_rate": pct(flaw_exact, len(flaw_rows)),
        "synthetic_flaws_broad_top2": flaw_top2,
        "synthetic_flaws_broad_top2_rate": pct(flaw_top2, len(flaw_rows)),
        "wrong_character_total": len(wrong_rows),
        "wrong_character_blocked": wrong_pass,
        "wrong_character_block_rate": pct(wrong_pass, len(wrong_rows)),
    }


def make_proof_image(summary: dict[str, Any], output_path: Path) -> None:
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 52)
        h_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30)
        b_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 23)
        r_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 21)
        s_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    except OSError:
        title_font = h_font = b_font = r_font = s_font = ImageFont.load_default()

    width, height = 1800, 1160
    bg = (248, 250, 252)
    ink = (15, 23, 42)
    muted = (71, 85, 105)
    border = (203, 213, 225)
    white = (255, 255, 255)
    green = (22, 163, 74)
    blue = (37, 99, 235)
    amber = (217, 119, 6)
    red = (220, 38, 38)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    draw.text((60, 42), "Deep Validation: Full 62-Character System", font=title_font, fill=ink)
    draw.text(
        (62, 108),
        "Validation covers assets, good samples, synthetic regional flaws, and wrong-character blocking.",
        font=r_font,
        fill=muted,
    )

    metrics = [
        ("Classes covered", f"{summary['classes_total']}/62", blue, "Every folder in data/Dataset"),
        ("Asset coverage", f"{summary['assets_pass']}/62", green, "Processed, augmented, reference, model, baseline"),
        (
            "Good samples clean",
            f"{summary['good_samples_pass']}/{summary['good_samples_total']}",
            green if summary["good_samples_pass_rate"] >= 90 else amber,
            "Correct samples with no fine/broad flags",
        ),
        (
            "Synthetic flaws #1",
            f"{summary['synthetic_flaws_broad_exact']}/{summary['synthetic_flaws_total']}",
            green if summary["synthetic_flaws_broad_exact_rate"] >= 70 else amber,
            "Intended top/middle/bottom is broad-band rank #1",
        ),
        (
            "Synthetic flaws top-2",
            f"{summary['synthetic_flaws_broad_top2']}/{summary['synthetic_flaws_total']}",
            green if summary["synthetic_flaws_broad_top2_rate"] >= 85 else amber,
            "Intended broad band appears in top 2",
        ),
        (
            "Wrong-char blocked",
            f"{summary['wrong_character_blocked']}/{summary['wrong_character_total']}",
            green if summary["wrong_character_block_rate"] >= 90 else red,
            "Mismatched selected class returns score 0",
        ),
    ]

    for index, (label, value, color, note) in enumerate(metrics):
        x = 60 + (index % 3) * 565
        y = 185 + (index // 3) * 205
        draw.rounded_rectangle((x, y, x + 520, y + 160), radius=20, fill=white, outline=border, width=2)
        draw.text((x + 24, y + 20), label, font=r_font, fill=muted)
        draw.text((x + 24, y + 58), value, font=h_font, fill=color)
        wrapped = textwrap.wrap(note, 44)
        ty = y + 108
        for line in wrapped:
            draw.text((x + 24, ty), line, font=s_font, fill=muted)
            ty += 22

    draw.rounded_rectangle((60, 625, 1740, 965), radius=24, fill=white, outline=border, width=2)
    draw.text((95, 660), "What This Proves For Final Defense", font=h_font, fill=ink)
    bullets = [
        "The final system is no longer limited to the 5 controlled demo characters.",
        "All 62 dataset classes have processed data, augmented data, references, recognizer support, autoencoders, and calibrated baselines.",
        "Good handwriting is checked for false positives, not only flawed samples.",
        "Synthetic controlled flaws test whether the region-feedback mechanism points to the intended broad area.",
        "Wrong-character blocking protects the app from giving high scores to a different selected glyph.",
    ]
    y = 720
    for bullet in bullets:
        draw.ellipse((98, y + 8, 110, y + 20), fill=blue)
        for line in textwrap.wrap(bullet, 118):
            draw.text((125, y), line, font=r_font, fill=ink)
            y += 30
        y += 8

    draw.line((60, 1040, 1740, 1040), fill=border, width=2)
    draw.text((60, 1065), f"Generated artifact: {output_path}", font=s_font, fill=muted)
    draw.text(
        (60, 1090),
        "Note: synthetic flaw validation is controlled digital validation. Human-drawn flawed samples remain the stronger evidence for the 5 demo characters.",
        font=s_font,
        fill=muted,
    )
    image.save(output_path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classes = dataset_classes()
    print(f"Classes found: {len(classes)}")
    print("Validating assets...")
    asset_rows = validate_assets(classes)
    print("Validating good samples...")
    good_rows = validate_good_samples(classes, args.good_samples_per_class, device)
    print("Validating synthetic broad-band flaws...")
    flaw_rows = validate_synthetic_flaws(
        classes,
        args.synthetic_samples_per_band,
        device,
        args.output_dir,
        args.synthetic_flaw_kind,
    )
    print("Validating wrong-character blocking...")
    wrong_rows = validate_wrong_character_blocking(classes, args.wrong_character_samples_per_class)

    summary = summarize(classes, asset_rows, good_rows, flaw_rows, wrong_rows)

    write_csv(args.output_dir / "asset_coverage_62.csv", asset_rows)
    write_csv(args.output_dir / "good_sample_validation_62.csv", good_rows)
    write_csv(args.output_dir / "synthetic_flaw_validation_62.csv", flaw_rows)
    write_csv(args.output_dir / "wrong_character_blocking_62.csv", wrong_rows)
    (args.output_dir / "deep_validation_summary_62.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    make_proof_image(summary, args.output_dir / "deep_validation_62_proof.png")

    print(json.dumps(summary, indent=2))
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
