"""Create transparent display glyphs from raw Ranjana reference photos."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.preprocessing.normalize import _binarize_ink, _find_ink_bbox, _load_grayscale  # noqa: E402


OUTPUT_SIZE = 256
PADDING_RATIO = 0.12


def create_display_glyph(reference_path: Path, output_path: Path, output_size: int = OUTPUT_SIZE) -> None:
    grayscale = _load_grayscale(reference_path)
    binary = _binarize_ink(grayscale)
    x, y, width, height = _find_ink_bbox(binary)
    cropped = binary[y : y + height, x : x + width]

    padding = max(4, int(max(width, height) * PADDING_RATIO))
    padded = cv2.copyMakeBorder(
        cropped,
        padding,
        padding,
        padding,
        padding,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    padded_height, padded_width = padded.shape[:2]
    scale = (output_size - 2) / float(max(padded_width, padded_height))
    resized_width = max(1, int(round(padded_width * scale)))
    resized_height = max(1, int(round(padded_height * scale)))
    resized = cv2.resize(padded, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    _, alpha = cv2.threshold(resized, 32, 255, cv2.THRESH_BINARY)

    rgba = np.zeros((output_size, output_size, 4), dtype=np.uint8)
    y_offset = (output_size - resized_height) // 2
    x_offset = (output_size - resized_width) // 2
    glyph = rgba[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width]
    glyph[..., 3] = alpha

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rgba)


def main() -> None:
    reference_root = PROJECT_ROOT / "data" / "Reference"
    output_root = BACKEND_ROOT / "ml" / "display_glyphs"
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Missing reference root: {reference_root}")

    created = 0
    skipped: list[str] = []
    for class_dir in sorted(path for path in reference_root.iterdir() if path.is_dir()):
        reference_path = class_dir / "photo-1" / f"{class_dir.name}.jpg"
        if not reference_path.is_file():
            skipped.append(f"{class_dir.name}: missing {reference_path}")
            continue

        output_path = output_root / f"{class_dir.name}.png"
        create_display_glyph(reference_path, output_path)
        created += 1

    print(f"Created display glyphs: {created}")
    print(f"Output root: {output_root}")
    if skipped:
        print("Skipped:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
